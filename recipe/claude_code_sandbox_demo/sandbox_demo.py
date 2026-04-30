import time

from env_gateway_sdk import EnvClient, GatewayClient

model_name = "xtuner_gateway_demo"
api_key = "sk-admin"
base_url = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"
nvm_src = "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh"

print("Creating gateway client...")
gateway_client = GatewayClient(base_url="http://env-gateway.ailab.ailab.ai")

print("\nCreating environment...")
env = gateway_client.create(image_tag="hb_citation-check", ttl_seconds=1800)
print(env.env_id, env.url)

print("\nCreating environment client...")
env_client = EnvClient(env.url)

print(env_client.wait_ready(timeout=120, interval=2))

print("\nStarting keepalive...")
env_client.keepalive(in_secs=10)
print("\nChecking health...")
print(env_client.health())

# Install nvm + Node.js LTS + claude-code as root (mirrors _install_claude_code in skillsbench_eval.py)
print("\nInstalling claude-code...")
start_time = time.time()

# Step 1: Install system deps (curl + bash needed by nvm installer)
print(env_client.exec(
    "sh -c 'set -e; "
    "if command -v apk >/dev/null 2>&1; then apk add --no-cache bash curl ca-certificates; "
    "elif command -v apt-get >/dev/null 2>&1; then apt-get update -qq && apt-get install -y -qq bash curl ca-certificates; "
    "elif command -v dnf >/dev/null 2>&1; then dnf install -y bash curl ca-certificates; "
    "elif command -v yum >/dev/null 2>&1; then yum install -y bash curl ca-certificates; "
    "fi'",
    timeout_sec=60,
))

# Step 2: Install nvm under /root/.nvm
print(env_client.exec(
    f"bash -c 'export NVM_DIR=\"/root/.nvm\" && curl -fsSL {nvm_src} | bash'",
    timeout_sec=300,
))

# Step 3: Install Node.js LTS
print(env_client.exec(
    "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && nvm install --lts'",
    timeout_sec=300,
))

# Step 4: Install @anthropic-ai/claude-code and verify
print(env_client.exec(
    "bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && "
    "npm install -g @anthropic-ai/claude-code && claude --version'",
    timeout_sec=300,
))

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print("========================claude -p 'Reply with exactly: pong'================================")
print(env_client.exec(
    f"bash -c 'export NVM_DIR=\"/root/.nvm\" && . \"$NVM_DIR/nvm.sh\" && "
    f"IS_SANDBOX=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 "
    f"ANTHROPIC_API_KEY={api_key} ANTHROPIC_BASE_URL={base_url} "
    f"ANTHROPIC_DEFAULT_SONNET_MODEL={model_name} "
    f"claude --permission-mode=bypassPermissions -p \"Reply with exactly: pong\"'"
))

# GatewayClient.close(env_id) closes the remote environment;
# EnvClient.close() stops the local keepalive background thread.
print("\nClosing environment client...")
env_client.close()
print("\nClosing environment...")
print(gateway_client.close(env.env_id))
