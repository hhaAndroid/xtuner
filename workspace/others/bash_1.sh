#!/bin/bash
# 将本地文件添加到 LFS 缓存

for file in $(git lfs ls-files | awk '{print $3}'); do
    # 获取 OID
    oid=$(cat "$file" | grep "oid sha256:" | cut -d: -f2 | tr -d ' ')

    # 如果本地有同名文件（真实内容）
    if [ -f "/path/to/your/local/files/$file" ]; then
        # 计算目录结构
        dir1=${oid:0:2}
        dir2=${oid:2:2}

        # 创建目录并复制
        mkdir -p .git/lfs/objects/$dir1/$dir2
        cp "/path/to/your/local/files/$file" .git/lfs/objects/$dir1/$dir2/$oid

        echo "Cached: $file"
    fi
done