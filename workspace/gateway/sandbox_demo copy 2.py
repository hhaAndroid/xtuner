from env_gateway_sdk import EnvClient, GatewayClient, HealthCheckError

import shlex
import time
print("Creating gateway client...")
gateway_client = GatewayClient(base_url="http://env-gateway.ailab.ailab.ai")

print("\nCreating environment...")
env = gateway_client.create(image_tag="hb_citation-check", ttl_seconds=1800)
print(env.env_id, env.url)

print("\nCreating environment client...")
env_client = EnvClient(env.url)

print(env_client.wait_ready(timeout=120, interval=2))

print("\nStarting keepalive...")
env_client.keepalive(in_secs=10)  # 立即返回，后台线程开始发送心跳
print("\nChecking health...")
print(env_client.health())
print("\nExecuting command...")
node_version = "22"
claude_code_version = ""  # 比如 "1.2.3"，留空则安装最新版

install_script = f"""
set -euo pipefail

NODE_VERSION="${{NODE_VERSION:-{node_version}}}"
CLAUDE_CODE_VERSION="${{CLAUDE_CODE_VERSION:-{claude_code_version}}}"

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"

if command -v claude >/dev/null 2>&1; then
    echo "Claude Code already installed: $(claude --version)"
    exit 0
fi

if command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl bash procps tar xz python3
elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq curl procps xz-utils python3
fi

echo "Installing Node.js v${{NODE_VERSION}}..."
NODE_MIRROR="${{NVM_NODEJS_ORG_MIRROR:-https://npmmirror.com/mirrors/node}}"
NODE_FULL_VERSION=$(curl -fsSL "${{NODE_MIRROR}}/index.json" | python3 -c "import sys,json; vs=[v['version'] for v in json.load(sys.stdin) if v['version'].startswith('v' + '${{NODE_VERSION}}' + '.')]; print(vs[0] if vs else 'v' + '${{NODE_VERSION}}' + '.0.0')")

echo "Resolved Node.js version: ${{NODE_FULL_VERSION}}"
cd /tmp
for i in 1 2 3; do
    if curl -fSL "${{NODE_MIRROR}}/${{NODE_FULL_VERSION}}/node-${{NODE_FULL_VERSION}}-linux-x64.tar.xz" -o node.tar.xz; then
        break
    fi
    if [ "$i" -eq 3 ]; then
        echo "Node.js download failed after retries"
        exit 1
    fi
    echo "Download attempt $i failed, retrying..."
    sleep 2
done

tar -xf node.tar.xz
cp -r node-${{NODE_FULL_VERSION}}-linux-x64/bin/* /usr/local/bin/
cp -r node-${{NODE_FULL_VERSION}}-linux-x64/lib/* /usr/local/lib/
rm -rf node.tar.xz node-${{NODE_FULL_VERSION}}-linux-x64

echo "Node.js installed: $(node --version)"
echo "npm installed: $(npm --version)"
npm config set registry https://registry.npmmirror.com

if [ -n "${{CLAUDE_CODE_VERSION}}" ]; then
    npm install -g "@anthropic-ai/claude-code@${{CLAUDE_CODE_VERSION}}"
else
    npm install -g @anthropic-ai/claude-code
fi

claude --version
"""

print(env_client.exec(f"bash -lc {shlex.quote(install_script)}"))
print(env_client.exec("bash -lc 'command -v claude && claude --version'"))

model_name = "xtuner_gateway_demo"
api_key = "sk-admin"
base_url = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"

print(env_client.exec(
    f"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 ANTHROPIC_API_KEY={api_key} ANTHROPIC_BASE_URL={base_url} ANTHROPIC_DEFAULT_SONNET_MODEL={model_name} "
    "claude -p 'Reply with exactly: pong'"
))

# GatewayClient.close(env_id) 负责关闭远端环境，而 EnvClient.close() 负责停止本地这个 EnvClient 持有的后台 keepalive 线程。
print("\nClosing environment client...")
env_client.close()
print("\nClosing environment...")
print(gateway_client.close(env.env_id))