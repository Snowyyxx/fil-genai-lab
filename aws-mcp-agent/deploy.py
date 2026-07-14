"""
Deploy the MCP agent to ECS Fargate — idempotent, boto3-only (no aws CLI needed).

THE TWO IDENTITIES (the whole point):
  - We run THIS script with the deployer IAM user's keys (from .env) to BUILD infra.
  - The running container uses a TASK ROLE (created here) to call Bedrock. The task
    definition contains NO AWS keys — ECS injects temporary creds from the role.

Run stages independently:  python deploy.py <stage>
  stages: ecr | roles | logs | cluster | taskdef | run | url | all
Re-running is safe (each step creates-or-reuses).
"""
from __future__ import annotations

import json
import sys
import time

import boto3
from dotenv import dotenv_values

CFG = dotenv_values(".env")
REGION = CFG["AWS_DEFAULT_REGION"]
KW = dict(
    aws_access_key_id=CFG["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=CFG["AWS_SECRET_ACCESS_KEY"],
    region_name=REGION,
)

# --- names / config -------------------------------------------------------
APP = "aws-mcp-agent"
ECR_REPO = APP
CLUSTER = "mcp-cluster"
TASK_FAMILY = "aws-mcp-agent-task"
SERVICE = "aws-mcp-agent-svc"
EXEC_ROLE = "mcpAgentExecutionRole"   # ECS agent: pull image, write logs
TASK_ROLE = "mcpAgentTaskRole"        # our app: call Bedrock
LOG_GROUP = f"/ecs/{APP}"
CONTAINER_PORT = 8000
MODEL_ID = "eu.amazon.nova-lite-v1:0"
SG_NAME = "mcp-agent-sg"

ACCOUNT = boto3.client("sts", **KW).get_caller_identity()["Account"]
ECR_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"

TRUST = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})


def log(msg): print(f"  {msg}")


# --- stages ---------------------------------------------------------------
def stage_ecr():
    ecr = boto3.client("ecr", **KW)
    try:
        ecr.create_repository(repositoryName=ECR_REPO)
        log(f"created ECR repo {ECR_REPO}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        log(f"ECR repo {ECR_REPO} already exists")
    print(f"ECR_URI={ECR_URI}")


def stage_roles():
    iam = boto3.client("iam", **KW)

    # Execution role: lets the ECS agent pull from ECR and push logs.
    _make_role(iam, EXEC_ROLE)
    iam.attach_role_policy(
        RoleName=EXEC_ROLE,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )
    log(f"execution role ready: {EXEC_ROLE}")

    # Task role: what OUR APP is allowed to do at runtime = invoke Bedrock.
    _make_role(iam, TASK_ROLE)
    iam.put_role_policy(
        RoleName=TASK_ROLE,
        PolicyName="BedrockInvoke",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                # cross-region inference profile + the underlying nova models
                "Resource": [
                    f"arn:aws:bedrock:*:{ACCOUNT}:inference-profile/eu.amazon.nova-*",
                    "arn:aws:bedrock:*::foundation-model/amazon.nova-*",
                ],
            }],
        }),
    )
    log(f"task role ready with bedrock:InvokeModel: {TASK_ROLE}")


def _make_role(iam, name):
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=TRUST)
        log(f"created role {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=TRUST)
        log(f"role {name} exists (trust refreshed)")


def stage_logs():
    logs = boto3.client("logs", **KW)
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        log(f"created log group {LOG_GROUP}")
    except logs.exceptions.ResourceAlreadyExistsException:
        log(f"log group {LOG_GROUP} exists")


def stage_cluster():
    ecs = boto3.client("ecs", **KW)
    ecs.create_cluster(clusterName=CLUSTER, capacityProviders=["FARGATE"])
    log(f"cluster ready: {CLUSTER}")


def stage_taskdef():
    ecs = boto3.client("ecs", **KW)
    resp = ecs.register_task_definition(
        family=TASK_FAMILY,
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu="512",
        memory="1024",
        executionRoleArn=f"arn:aws:iam::{ACCOUNT}:role/{EXEC_ROLE}",
        taskRoleArn=f"arn:aws:iam::{ACCOUNT}:role/{TASK_ROLE}",  # <-- app's Bedrock identity
        containerDefinitions=[{
            "name": APP,
            "image": f"{ECR_URI}:latest",
            "essential": True,
            "portMappings": [{"containerPort": CONTAINER_PORT, "protocol": "tcp"}],
            # NO AWS keys here — only non-secret config. Creds come from the task role.
            "environment": [
                {"name": "AWS_DEFAULT_REGION", "value": REGION},
                {"name": "BEDROCK_MODEL_ID", "value": MODEL_ID},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": LOG_GROUP,
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "ecs",
                },
            },
        }],
    )
    rev = resp["taskDefinition"]["taskDefinitionArn"]
    log(f"registered task def: {rev}")


def _default_network():
    """Find the default VPC's subnets and ensure a security group open on 8000."""
    ec2 = boto3.client("ec2", **KW)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]
    subnets = [s["SubnetId"] for s in ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]]

    # security group: create-or-get, allow inbound CONTAINER_PORT from anywhere
    sgs = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SG_NAME]},
        {"Name": "vpc-id", "Values": [vpc_id]}])["SecurityGroups"]
    if sgs:
        sg_id = sgs[0]["GroupId"]
    else:
        sg_id = ec2.create_security_group(
            GroupName=SG_NAME, Description="MCP agent public 8000", VpcId=vpc_id)["GroupId"]
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": CONTAINER_PORT, "ToPort": CONTAINER_PORT,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "public"}],
                }])
        except Exception as e:
            log(f"ingress rule note: {e}")
    return subnets, sg_id


def stage_run():
    ecs = boto3.client("ecs", **KW)
    subnets, sg_id = _default_network()
    log(f"subnets={subnets[:2]}... sg={sg_id}")
    net = {"awsvpcConfiguration": {
        "subnets": subnets,
        "securityGroups": [sg_id],
        "assignPublicIp": "ENABLED",  # <-- gives the task a public IP (no load balancer)
    }}
    # create-or-update the service (desiredCount=1)
    existing = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"]
    active = [s for s in existing if s["status"] == "ACTIVE"]
    if active:
        ecs.update_service(cluster=CLUSTER, service=SERVICE, taskDefinition=TASK_FAMILY,
                           desiredCount=1, forceNewDeployment=True)
        log(f"updated service {SERVICE}")
    else:
        ecs.create_service(
            cluster=CLUSTER, serviceName=SERVICE, taskDefinition=TASK_FAMILY,
            desiredCount=1, launchType="FARGATE", networkConfiguration=net)
        log(f"created service {SERVICE}")


def stage_url():
    """Wait for a running task and print its public IP + curl command."""
    ecs = boto3.client("ecs", **KW)
    ec2 = boto3.client("ec2", **KW)
    for _ in range(40):
        tasks = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE,
                               desiredStatus="RUNNING")["taskArns"]
        if tasks:
            d = ecs.describe_tasks(cluster=CLUSTER, tasks=tasks)["tasks"][0]
            eni = next((kv["value"] for a in d.get("attachments", [])
                        for kv in a["details"] if kv["name"] == "networkInterfaceId"), None)
            if eni:
                ni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni])["NetworkInterfaces"][0]
                ip = ni.get("Association", {}).get("PublicIp")
                if ip:
                    print(f"PUBLIC_IP={ip}")
                    print(f"  curl http://{ip}:{CONTAINER_PORT}/health")
                    print(f"  curl -X POST http://{ip}:{CONTAINER_PORT}/solve "
                          f"-H 'Content-Type: application/json' "
                          f"-d '{{\"problem\":\"12*12+5\",\"target_solutions\":1}}'")
                    return ip
        log("waiting for RUNNING task with public IP...")
        time.sleep(15)
    log("timed out waiting for a running task (check ECS console / logs)")


STAGES = {
    "ecr": stage_ecr, "roles": stage_roles, "logs": stage_logs,
    "cluster": stage_cluster, "taskdef": stage_taskdef, "run": stage_run, "url": stage_url,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["ecr", "roles", "logs", "cluster", "taskdef", "run", "url"]
    todo = order if which == "all" else [which]
    print(f"account={ACCOUNT} region={REGION}")
    for name in todo:
        print(f"\n== {name} ==")
        STAGES[name]()


if __name__ == "__main__":
    main()
