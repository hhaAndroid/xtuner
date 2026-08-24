from env_gateway_sdk import EnvClient, GatewayClient, HealthCheckError

import time
print("Creating gateway client...")
gateway_client = GatewayClient(base_url="http://env-gateway.ailab.ailab.ai", key="huangha-kdio28HD")

print("\nCreating environment...")
env = gateway_client.create(image_tag="t-data-processing-v1", ttl_seconds=1800)
print(env.env_id, env.url)

print("\nCreating environment client...")
env_client = EnvClient(env.url)

print(env_client.wait_ready(timeout=120, interval=2))

print("\nStarting keepalive...")
env_client.keepalive(in_secs=10)  # 立即返回，后台线程开始发送心跳
print("\nChecking health...")
print(env_client.health())
print("\nExecuting command...")
# Step 2: 安装 nvm
print(env_client.exec("bash -lc 'curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash'"))
# Step 4: 安装 Node LTS（含 npm）
print(env_client.exec("bash -lc 'export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && nvm install --lts'"))
# Step 6: 安装 claude-code
print(env_client.exec("bash -lc 'export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && npm install -g @anthropic-ai/claude-code'"))
# Step 7: 将 claude 暴露到系统 PATH（后续可直接 env_client.exec(\"claude ...\")）
print(env_client.exec("bash -lc 'export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" && nvm alias default \"lts/*\" && CLAUDE_BIN=$(command -v claude) && ln -sf \"$CLAUDE_BIN\" /usr/local/bin/claude && command -v claude && claude --version'"))
# Step 8: 直接调用验证（不再需要每次 source nvm）
print(env_client.exec("claude --version"))


api_key = "sk-admin"
base_url = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"

print(env_client.exec(
    f"ANTHROPIC_API_KEY={api_key} ANTHROPIC_BASE_URL={base_url} "
    "claude --print -p 'What is the weather in Tokyo?'"
))

# GatewayClient.close(env_id) 负责关闭远端环境，而 EnvClient.close() 负责停止本地这个 EnvClient 持有的后台 keepalive 线程。
print("\nClosing environment client...")
env_client.close()
print("\nClosing environment...")
print(gateway_client.close(env.env_id))