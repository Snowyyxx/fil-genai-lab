"""
Upload the frontend HTML to S3 so the app can fetch it at runtime.

The website HTML is public content, so we serve it from a public S3 object and
the app fetches it over plain HTTPS (no AWS creds needed in the app). This
requires S3 permissions on the DEPLOYER user (attach AmazonS3FullAccess) — the
account currently denies S3, so run this after attaching that policy.

Prints HTML_URL — set that env var on the app (locally or in the task def).

    python upload_html_to_s3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
from dotenv import dotenv_values

HERE = Path(__file__).parent
CREDS = dotenv_values(HERE.parent / "aws-mcp-agent" / ".env")
REGION = CREDS["AWS_DEFAULT_REGION"]
KW = dict(aws_access_key_id=CREDS["AWS_ACCESS_KEY_ID"],
          aws_secret_access_key=CREDS["AWS_SECRET_ACCESS_KEY"], region_name=REGION)

ACCOUNT = boto3.client("sts", **KW).get_caller_identity()["Account"]
BUCKET = f"faraz-shayari-web-{ACCOUNT}"
KEY = "frontend.html"


def main() -> None:
    s3 = boto3.client("s3", **KW)

    # 1. bucket (create if missing)
    try:
        s3.create_bucket(Bucket=BUCKET,
                         CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"created bucket {BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"bucket {BUCKET} exists")

    # 2. allow public reads (the HTML is public web content)
    s3.delete_public_access_block(Bucket=BUCKET)
    s3.put_bucket_policy(Bucket=BUCKET, Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "PublicReadHTML", "Effect": "Allow", "Principal": "*",
            "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{BUCKET}/*",
        }]}))

    # 3. upload the HTML
    s3.put_object(Bucket=BUCKET, Key=KEY,
                  Body=(HERE / "frontend.html").read_bytes(),
                  ContentType="text/html; charset=utf-8")

    url = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{KEY}"
    print(f"\n✓ uploaded. Set this on the app:\n  HTML_URL={url}")


if __name__ == "__main__":
    main()
