import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_rope(dim=64, rope_theta=10000, max_seq_length=512, num_dims_to_plot=3, 
                   output_prefix='rope'):
    """
    可视化 RoPE 的频率曲线
    
    参数:
        dim: 嵌入维度
        rope_theta: RoPE 的基础频率参数
        max_seq_length: 最大序列长度
        num_dims_to_plot: 要绘制的维度数量
        output_prefix: 输出文件名前缀
    """
    
    # 计算每个维度对的频率
    # freq_i = 1 / (theta ^ (2i/dim)) for i in [0, dim/2)
    # RoPE 是成对工作的：(dim_0, dim_1), (dim_2, dim_3), ..., (dim_62, dim_63)
    num_pairs = dim // 2
    pair_indices = np.arange(num_pairs)  # 0, 1, 2, ..., 31 (for dim=64)
    freqs = 1.0 / (rope_theta ** (2 * pair_indices / dim))
    
    # 计算周期
    periods = 2 * np.pi / freqs
    
    # 找到最大周期（最后一个维度对，频率最低）
    max_period = periods[-1]
    min_freq = freqs[-1]
    
    # 生成位置序列
    positions = np.arange(max_seq_length)
    
    # 打印周期信息
    print(f"\n{'='*80}")
    print(f"RoPE 周期分析 - {output_prefix}")
    print(f"{'='*80}")
    print(f"总维度数 (dim): {dim}")
    print(f"维度对数量: {num_pairs} 对 (每对2个维度共享同一频率)")
    print(f"Theta: {rope_theta}")
    print(f"可视化序列长度: {max_seq_length}")
    print(f"\n💡 说明：RoPE 中每2个维度组成一对，应用相同频率的旋转")
    print(f"   例如：Dim[0,1] 共享频率0, Dim[2,3] 共享频率1, ..., Dim[{dim-2},{dim-1}] 共享频率{num_pairs-1}")
    print(f"\n{'维度对':>10} | {'频率索引':>8} | {'频率值':>12} | {'周期(positions)':>18}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*18}")
    
    # 打印前5个维度对
    for i in range(min(5, num_pairs)):
        dim_pair = f"[{i*2:2d},{i*2+1:2d}]"
        print(f"{dim_pair:>10} | {i:>8d} | {freqs[i]:12.8f} | {periods[i]:18.2f}")
    
    if num_pairs > 10:
        print(f"{'...':>10} | {'...':>8} | {'...':>12} | {'...':>18}")
        # 打印最后5个维度对
        for i in range(max(5, num_pairs-5), num_pairs):
            dim_pair = f"[{i*2:2d},{i*2+1:2d}]"
            print(f"{dim_pair:>10} | {i:>8d} | {freqs[i]:12.8f} | {periods[i]:18.2f}")
    elif num_pairs > 5:
        for i in range(5, num_pairs):
            dim_pair = f"[{i*2:2d},{i*2+1:2d}]"
            print(f"{dim_pair:>10} | {i:>8d} | {freqs[i]:12.8f} | {periods[i]:18.2f}")
    
    print(f"\n{'='*80}")
    print(f"⭐ 关键信息：")
    print(f"{'='*80}")
    print(f"最高频率 (Dim[0,1]):        {freqs[0]:.8f}")
    print(f"最低频率 (Dim[{dim-2},{dim-1}]):      {freqs[-1]:.8f}")
    print(f"\n最短周期 (Dim[0,1]):        {periods[0]:.2f} positions")
    print(f"最长周期 (Dim[{dim-2},{dim-1}]):      {periods[-1]:.2f} positions")
    print(f"\n🔄 触发所有维度周期重复所需的序列长度: {max_period:.2f} positions")
    print(f"   (即最后一个维度对 Dim[{dim-2},{dim-1}] 旋转一周所需的长度)")
    
    if max_seq_length < max_period:
        print(f"\n⚠️  当前可视化长度 ({max_seq_length}) < 最长周期 ({max_period:.2f})")
        print(f"   最后的维度对在当前序列中只完成了 {max_seq_length/max_period*100:.1f}% 的周期")
    else:
        print(f"\n✓ 当前可视化长度 ({max_seq_length}) >= 最长周期 ({max_period:.2f})")
        print(f"  最后的维度对在当前序列中完成了 {max_seq_length/max_period:.2f} 个完整周期")
    
    # 计算频率比（最高频率/最低频率）
    freq_ratio = freqs[0] / freqs[-1]
    print(f"\n📊 频率跨度: {freq_ratio:.2f}x (最高频/最低频)")
    print(f"{'='*80}\n")
    
    # 创建图形
    fig, axes = plt.subplots(num_dims_to_plot, 1, figsize=(12, 3*num_dims_to_plot))
    if num_dims_to_plot == 1:
        axes = [axes]
    
    # 为不同的维度对绘制曲线
    pairs_to_visualize = np.linspace(0, num_pairs-1, num_dims_to_plot, dtype=int)
    
    for idx, (ax, pair_idx) in enumerate(zip(axes, pairs_to_visualize)):
        freq = freqs[pair_idx]
        period = periods[pair_idx]
        # 计算该维度在不同位置的角度值
        angles = positions * freq
        # 绘制正弦曲线（RoPE 使用 sin 和 cos）
        values = np.sin(angles)
        
        ax.plot(positions, values, linewidth=1.5, color='#4A90E2')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        
        dim_range = f"[{pair_idx*2},{pair_idx*2+1}]"
        ax.set_title(f'Dimension Pair {dim_range}/{dim} | Frequency: {freq:.6f} | Period: {period:.1f} positions', 
                    fontsize=12, pad=10)
        ax.set_xlim(0, max_seq_length)
        ax.set_ylim(-1.1, 1.1)
        
        # 标记完整周期
        num_complete_cycles = int(max_seq_length / period)
        if num_complete_cycles > 0 and num_complete_cycles < 20:  # 避免标记太多
            for cycle in range(1, num_complete_cycles + 1):
                cycle_pos = cycle * period
                if cycle_pos <= max_seq_length:
                    ax.axvline(x=cycle_pos, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    waveform_filename = f'{output_prefix}_waveforms.png'
    plt.savefig(waveform_filename, dpi=300, bbox_inches='tight')
    print(f"📁 波形图已保存为 {waveform_filename}")
    plt.close()
    
    # 额外绘制：所有维度的频率分布
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    pair_dim_indices = pair_indices * 2  # 转换为实际维度索引显示
    ax2.plot(pair_dim_indices, freqs, marker='o', linewidth=2, markersize=4, color='#4A90E2', label='Frequency')
    ax2.set_xlabel('Dimension Index (每对的第一个维度)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'RoPE Frequencies across Dimension Pairs (theta={rope_theta}, {num_pairs} pairs)', 
                 fontsize=14, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    plt.tight_layout()
    freq_filename = f'{output_prefix}_frequencies.png'
    plt.savefig(freq_filename, dpi=300, bbox_inches='tight')
    print(f"📁 频率分布图已保存为 {freq_filename}")
    plt.close()
    
    # 绘制周期分布图
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ax3.plot(pair_dim_indices, periods, marker='s', linewidth=2, markersize=4, color='#E94B3C', label='Period')
    ax3.axhline(y=max_seq_length, color='green', linestyle='--', linewidth=2, 
                label=f'Current seq_length ({max_seq_length})')
    ax3.axhline(y=max_period, color='orange', linestyle='--', linewidth=2, 
                label=f'Max period ({max_period:.1f})')
    ax3.set_xlabel('Dimension Index (每对的第一个维度)', fontsize=12)
    ax3.set_ylabel('Period (positions)', fontsize=12)
    ax3.set_title(f'RoPE Periods across Dimension Pairs ({num_pairs} pairs)', fontsize=14, pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.legend()
    plt.tight_layout()
    period_filename = f'{output_prefix}_periods.png'
    plt.savefig(period_filename, dpi=300, bbox_inches='tight')
    print(f"📁 周期分布图已保存为 {period_filename}\n")
    plt.close()
    
    return {
        'dim': dim,
        'theta': rope_theta,
        'max_seq_length': max_seq_length,
        'max_period': max_period,
        'min_period': periods[0],
        'max_freq': freqs[0],
        'min_freq': freqs[-1],
        'num_pairs': num_pairs,
        'all_periods': periods,
        'all_freqs': freqs
    }

# 使用示例
if __name__ == "__main__":
    results = []
    
    # 对比不同的 theta 值
    print("\n" + "🔵" * 40)
    print("实验 1: 标准配置")
    print("🔵" * 40)
    result1 = visualize_rope(dim=64, rope_theta=10000, max_seq_length=512, 
                             num_dims_to_plot=3, output_prefix='rope_dim64_theta10000')
    results.append(result1)
    
    print("\n" + "🔵" * 40)
    print("实验 2: 长上下文配置")
    print("🔵" * 40)
    result2 = visualize_rope(dim=64, rope_theta=50000, max_seq_length=2048, 
                             num_dims_to_plot=3, output_prefix='rope_dim64_theta50000')
    results.append(result2)
    
    print("\n" + "🔵" * 40)
    print("实验 3: 短上下文配置")
    print("🔵" * 40)
    result3 = visualize_rope(dim=64, rope_theta=1000, max_seq_length=512, 
                             num_dims_to_plot=3, output_prefix='rope_dim64_theta1000')
    results.append(result3)
    
    print("\n" + "🔵" * 40)
    print("实验 4: 高维度配置")
    print("🔵" * 40)
    result4 = visualize_rope(dim=128, rope_theta=10000, max_seq_length=512, 
                             num_dims_to_plot=3, output_prefix='rope_dim128_theta10000')
    results.append(result4)
    
    print("\n" + "🔵" * 40)
    print("实验 5: LLaMA风格长上下文")
    print("🔵" * 40)
    result5 = visualize_rope(dim=128, rope_theta=500000, max_seq_length=8192, 
                             num_dims_to_plot=3, output_prefix='rope_dim128_theta500000')
    results.append(result5)
    
    print("\n" + "🔵" * 40)
    print("实验 6: Qwen3 30b 风格长上下文")
    print("🔵" * 40)
    result6 = visualize_rope(dim=128, rope_theta=1000000, max_seq_length=40960, 
                             num_dims_to_plot=3, output_prefix='rope_dim128_theta1000000')
    results.append(result6)

    print("\n" + "🔵" * 40)
    print("实验 6: Qwen3 30b think 风格长上下文")
    print("🔵" * 40)
    result7 = visualize_rope(dim=128, rope_theta=10000000, max_seq_length=262144, 
                             num_dims_to_plot=3, output_prefix='rope_dim128_theta10000000')
    results.append(result7)

    print("\n" + "🔵" * 40)
    print("实验 6: Qwen3vl 30b think 风格长上下文")
    print("🔵" * 40)
    result8 = visualize_rope(dim=128, rope_theta=5000000, max_seq_length=262144, 
                             num_dims_to_plot=3, output_prefix='rope_dim128_theta5000000')
    results.append(result8)


    # 总结对比
    print("\n" + "="*100)
    print("📊 所有实验对比总结")
    print("="*100)
    print(f"{'Dim':>5} | {'Theta':>10} | {'Seq Len':>8} | {'维度对数':>8} | {'最长周期':>12} | {'频率跨度':>12} | {'覆盖率':>10}")
    print("-"*100)
    for result in results:
        freq_span = result['max_freq'] / result['min_freq']
        coverage = result['max_seq_length'] / result['max_period'] * 100
        print(f"{result['dim']:>5d} | {result['theta']:>10,d} | {result['max_seq_length']:>8,d} | "
              f"{result['num_pairs']:>8d} | {result['max_period']:>12,.2f} | "
              f"{freq_span:>11,.2f}x | {coverage:>9.1f}%")
    print("="*100)
    print("\n说明：")
    print("  - Dim: 嵌入维度")
    print("  - Theta: RoPE基础频率参数")
    print("  - Seq Len: 可视化的序列长度")
    print("  - 维度对数: dim/2，每对共享一个频率")
    print("  - 最长周期: 最后一个维度对完成一个完整旋转所需的位置数")
    print("  - 频率跨度: 最高频率/最低频率的比值")
    print("  - 覆盖率: 当前序列长度占最长周期的百分比（>100%表示完成了多个周期）")
    print("="*100 + "\n")