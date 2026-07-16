import math

from deeparguing.md_log import write_markdown_log

# x = [17, 162, 2265, 50, 2680, 1297, 250, 354, 454, 44, 1094, 2891, 127, 312, 1395, 1815, 911, 1837]
x = [55, 25]

x.reverse()

total = sum(x)

lines = []
for v in x:
    w = math.sqrt(total/(len(x) * v))
    # w = (total/(len(x) * v))

    print("\t- type: value")
    print(f"\t  value: {w}")
    lines.append(f"- type: value\n  value: {w}")

write_markdown_log(
    ["--- WEIGHTS ---", "```yaml\n" + "\n".join(lines) + "\n```"],
    "outputs/logs/weights.md",
    mode="w",
)
