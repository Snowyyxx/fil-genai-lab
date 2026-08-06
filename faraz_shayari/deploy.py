"""
Deploy Faraz Shayari to ECS Fargate — idempotent, boto3-only.

Same two-identity model as aws-mcp-agent: this script runs as the deployer IAM
user (keys from ../aws-mcp-agent/.env) to build infra; the running container
calls Bedrock via the TASK ROLE (no keys in the image).

Stages:  python deploy.py <ecr|roles|logs|cluster|taskdef|run|url|all>
"""
from __future__ import annotations

import json
import sys
import time

import boto3
from dotenv import dotenv_values

CFG = dotenv_values("../aws-mcp-agent/.env")
REGION = CFG["AWS_DEFAULT_REGION"]
KW = dict(aws_access_key_id=CFG["AWS_ACCESS_KEY_ID"],
          aws_secret_access_key=CFG["AWS_SECRET_ACCESS_KEY"], region_name=REGION)

APP = "faraz-shayari"
ECR_REPO = APP
CLUSTER = "faraz-cluster"
TASK_FAMILY = "faraz-shayari-task"
SERVICE = "faraz-shayari-svc"
EXEC_ROLE = "farazExecutionRole"
TASK_ROLE = "farazTaskRole"
LOG_GROUP = f"/ecs/{APP}"
PORT = 8000
SG_NAME = "faraz-sg"
GEN_MODEL = "bedrock/eu.amazon.nova-pro-v1:0"

ACCOUNT = boto3.client("sts", **KW).get_caller_identity()["Account"]
ECR_URI = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}"
TRUST = json.dumps({"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"},
    "Action": "sts:AssumeRole"}]})


def log(m): print(f"  {m}")


def stage_ecr():
    ecr = boto3.client("ecr", **KW)
    try:
        ecr.create_repository(repositoryName=ECR_REPO); log("created ECR repo")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        log("ECR repo exists")
    print(f"ECR_URI={ECR_URI}")


def _make_role(iam, name):
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=TRUST); log(f"created {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=TRUST); log(f"{name} exists")


def stage_roles():
    iam = boto3.client("iam", **KW)
    _make_role(iam, EXEC_ROLE)
    iam.attach_role_policy(RoleName=EXEC_ROLE,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy")
    log("exec role ready")
    _make_role(iam, TASK_ROLE)
    iam.put_role_policy(RoleName=TASK_ROLE, PolicyName="BedrockInvoke",
        PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            "Resource": [   # Titan (embed) + Nova (generate) are both amazon.*
                "arn:aws:bedrock:*::foundation-model/amazon.*",
                f"arn:aws:bedrock:*:{ACCOUNT}:inference-profile/eu.amazon.*",
            ]}]}))
    log("task role ready (bedrock:InvokeModel on amazon.*)")


def stage_logs():
    logs = boto3.client("logs", **KW)
    try:
        logs.create_log_group(logGroupName=LOG_GROUP); log("created log group")
    except logs.exceptions.ResourceAlreadyExistsException:
        log("log group exists")


def stage_cluster():
    boto3.client("ecs", **KW).create_cluster(clusterName=CLUSTER, capacityProviders=["FARGATE"])
    log(f"cluster ready: {CLUSTER}")


def stage_taskdef():
    ecs = boto3.client("ecs", **KW)
    r = ecs.register_task_definition(
        family=TASK_FAMILY, requiresCompatibilities=["FARGATE"], networkMode="awsvpc",
        cpu="512", memory="1024",
        executionRoleArn=f"arn:aws:iam::{ACCOUNT}:role/{EXEC_ROLE}",
        taskRoleArn=f"arn:aws:iam::{ACCOUNT}:role/{TASK_ROLE}",
        containerDefinitions=[{
            "name": APP, "image": f"{ECR_URI}:latest", "essential": True,
            "portMappings": [{"containerPort": PORT, "protocol": "tcp"}],
            "environment": [
                {"name": "AWS_DEFAULT_REGION", "value": REGION},
                {"name": "GEN_MODEL", "value": GEN_MODEL},
            ],
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": LOG_GROUP, "awslogs-region": REGION,
                "awslogs-stream-prefix": "ecs"}},
        }])
    log(f"registered {r['taskDefinition']['taskDefinitionArn']}")


def _network():
    ec2 = boto3.client("ec2", **KW)
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnets = [s["SubnetId"] for s in ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]]
    sgs = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if sgs:
        sg = sgs[0]["GroupId"]
    else:
        sg = ec2.create_security_group(GroupName=SG_NAME, Description="faraz public 8000", VpcId=vpc)["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    return subnets, sg


def stage_run():
    ecs = boto3.client("ecs", **KW)
    subnets, sg = _network()
    net = {"awsvpcConfiguration": {"subnets": subnets, "securityGroups": [sg], "assignPublicIp": "ENABLED"}}
    active = [s for s in ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"]
              if s["status"] == "ACTIVE"]
    if active:
        ecs.update_service(cluster=CLUSTER, service=SERVICE, taskDefinition=TASK_FAMILY,
                           desiredCount=1, forceNewDeployment=True); log("service updated")
    else:
        ecs.create_service(cluster=CLUSTER, serviceName=SERVICE, taskDefinition=TASK_FAMILY,
                           desiredCount=1, launchType="FARGATE", networkConfiguration=net)
        log("service created")


def stage_url():
    ecs = boto3.client("ecs", **KW); ec2 = boto3.client("ec2", **KW)
    for _ in range(40):
        tasks = ecs.list_tasks(cluster=CLUSTER, serviceName=SERVICE, desiredStatus="RUNNING")["taskArns"]
        if tasks:
            d = ecs.describe_tasks(cluster=CLUSTER, tasks=tasks)["tasks"][0]
            eni = next((kv["value"] for a in d.get("attachments", []) for kv in a["details"]
                        if kv["name"] == "networkInterfaceId"), None)
            if eni:
                ni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni])["NetworkInterfaces"][0]
                ip = ni.get("Association", {}).get("PublicIp")
                if ip:
                    print(f"PUBLIC_IP={ip}")
                    print(f"  open  http://{ip}:{PORT}/")
                    return ip
        log("waiting for RUNNING task..."); time.sleep(15)
    log("timed out")


STAGES = {"ecr": stage_ecr, "roles": stage_roles, "logs": stage_logs,
          "cluster": stage_cluster, "taskdef": stage_taskdef, "run": stage_run, "url": stage_url}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["ecr", "roles", "logs", "cluster", "taskdef", "run", "url"]
    print(f"account={ACCOUNT} region={REGION}")
    for name in (order if which == "all" else [which]):
        print(f"\n== {name} =="); STAGES[name]()


if __name__ == "__main__":
    main()
