from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "to_human" / "weighted_4_2_1_certificate_plan_20260902.pdf"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
FONT_BOLD = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

pdfmetrics.registerFont(TTFont("CJK", FONT))
pdfmetrics.registerFont(TTFont("CJK-Bold", FONT_BOLD))

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#0F766E")
LIGHT_TEAL = colors.HexColor("#E7F5F3")
LIGHT_BLUE = colors.HexColor("#EAF1F8")
LIGHT_GRAY = colors.HexColor("#F4F6F8")
MID_GRAY = colors.HexColor("#64748B")
DARK = colors.HexColor("#17212B")
RED = colors.HexColor("#B42318")
AMBER = colors.HexColor("#B26A00")
GREEN = colors.HexColor("#18794E")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CJKTitle",
    fontName="CJK-Bold",
    fontSize=22,
    leading=30,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="CJKSubtitle",
    fontName="CJK",
    fontSize=10.5,
    leading=16,
    textColor=MID_GRAY,
    spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="CJKH1",
    fontName="CJK-Bold",
    fontSize=15,
    leading=21,
    textColor=NAVY,
    spaceBefore=10,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="CJKH2",
    fontName="CJK-Bold",
    fontSize=11.5,
    leading=17,
    textColor=TEAL,
    spaceBefore=7,
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="CJKBody",
    fontName="CJK",
    fontSize=9.3,
    leading=15,
    textColor=DARK,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="CJKSmall",
    fontName="CJK",
    fontSize=8,
    leading=12,
    textColor=DARK,
))
styles.add(ParagraphStyle(
    name="CJKCallout",
    fontName="CJK-Bold",
    fontSize=10,
    leading=16,
    textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="CJKFoot",
    fontName="CJK",
    fontSize=7.5,
    leading=10,
    textColor=MID_GRAY,
    alignment=TA_CENTER,
))


def P(text, style="CJKBody"):
    return Paragraph(text, styles[style])


def bullet(text):
    return Paragraph(f"• {text}", styles["CJKBody"])


def table(data, widths, header=True, font_size=8.2):
    wrapped = []
    for r, row in enumerate(data):
        wrapped.append([
            cell if hasattr(cell, "wrap") else Paragraph(str(cell), ParagraphStyle(
                name=f"cell-{r}",
                parent=styles["CJKSmall"],
                fontName="CJK-Bold" if header and r == 0 else "CJK",
                fontSize=font_size,
                leading=font_size + 3.2,
                textColor=colors.white if header and r == 0 else DARK,
            ))
            for cell in row
        ])
    t = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ]
        for r in range(1, len(data)):
            if r % 2 == 0:
                commands.append(("BACKGROUND", (0, r), (-1, r), LIGHT_GRAY))
    t.setStyle(TableStyle(commands))
    return t


def callout(title, body, color=TEAL, bg=LIGHT_TEAL):
    cell = [P(title, "CJKCallout"), Spacer(1, 2), P(body, "CJKBody")]
    t = Table([[cell]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.8, color),
        ("LINEBEFORE", (0, 0), (0, -1), 4, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 17 * mm, PAGE_W - 20 * mm, 17 * mm)
    canvas.setFont("CJK", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(20 * mm, 11 * mm, "Livestock-Agent · Weighted 4:2:1 MILP")
    canvas.drawRightString(PAGE_W - 20 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=17 * mm,
        bottomMargin=23 * mm,
        title="统一加权 4:2:1 MILP：当前证书规划、建模与中国求解诊断",
        author="Hermes Agent / Livestock-Agent",
        subject="MILP certificate plan and China root-LP diagnosis",
    )
    story = []

    story += [
        P("统一加权 4:2:1 MILP", "CJKTitle"),
        P("当前证书规划、建模说明与中国全零解诊断 · 2026-09-02 10:58 CST", "CJKSubtitle"),
        callout(
            "结论",
            "中国并不是被模型判定为“只能零转移”，也不是求解器已经证明全零最优。SCIP 只是在 39.5 秒通过 trivial heuristic 保存了天然可行的全零解；此后约 43 小时持续求解一个预处理后仍含 897 万变量的根节点 LP，尚未完成第一次 LP 松弛，因此还没有 LP dual bound、分支节点或后续启发式所需的信息。当前证书进度仍是“有可行下界 0，但没有可用上界”，相对 gap 尚不可定义。",
            color=RED,
            bg=colors.HexColor("#FFF1F0"),
        ),
        Spacer(1, 10),
        P("1. 现在能确定什么", "CJKH1"),
        table([
            ["检查项", "现场证据", "判断"],
            ["进程是否存活", "PID 3740291，连续约 43 小时 100% 单核 CPU", "存活，正在计算"],
            ["内存是否不足", "RSS 约 68.4 GiB；系统可用约 399–405 GiB；VmSwap=0", "不是 OOM/换页瓶颈"],
            ["是否进入分支定界", "日志没有 node table、LP iteration、dual bound；nodes 尚不可读", "尚未完成根 LP"],
            ["当前解", "trivial heuristic：objective 0.0", "仅是全零可行解，不是最优结论"],
            ["当前证书", "无非零 incumbent；无有效 dual bound；gap 不可计算", "未达到 0.1% 证书"],
            ["调试边界", "系统 ptrace/perf 策略禁止附加 gdb/strace/perf", "阶段判断基于日志、CPU/RSS、线程和历史对照"],
        ], [32 * mm, 86 * mm, 52 * mm]),
        Spacer(1, 8),
        P("核心解释", "CJKH2"),
        bullet("模型允许所有移动变量都取 0；因此全零解天然满足源库存、源/目的地氮、目的地氨和非负约束。SCIP 的 trivial heuristic 会立即保存它作为第一个 incumbent。"),
        bullet("这只是最大化问题的一个很弱下界。由于目标中 N 和环境项为正，存在可行转移时，非零解通常会优于 0；但求解器需要根 LP 或专门启发式才能构造并证明这些解。"),
        bullet("中国预处理后仍有 8,970,252 个变量，其中 8,959,443 个整数变量、8,813 个构成连续变量；根 LP 本身就是超大稀疏线性规划。"),
        bullet("64 个线程中只有主线程积累了显著 CPU 时间，其余线程大多等待。这一 SCIP/LP 路径在当前构建中实质为单线程，所以更多 CPU 核没有转化为根 LP 加速。"),
        PageBreak(),
    ]

    story += [
        P("2. 为什么中国卡在根 LP", "CJKH1"),
        table([
            ["因素", "中国现场量级", "影响"],
            ["稠密路线张量", "10,654,593 个允许的 source×destination×species 整数变量", "每个变量同时进入多类容量和目标表达式"],
            ["预处理后规模", "8,970,252 个变量、26,318 个约束", "删除 169 万变量后仍远超 EU/澳大利亚"],
            ["估算非零系数", "约 6,396 万个约束非零项（结构性估算）", "根 LP 矩阵构造、缩放、因子分解和迭代昂贵"],
            ["额外构成目标", "8,813 个连续变量；结构约束使行数由旧 primary-only 的 11,362 增至 26,318", "比仅最大化 N 的中国根 LP更重"],
            ["内存", "当前 RSS 68.4 GiB；HWM 69.0 GiB；无 swap", "内存稳定，但大型 LP 数据结构长期驻留"],
            ["单核求解", "主线程约 100%；其余线程等待", "43 小时墙钟≈43 小时有效单核计算"],
        ], [34 * mm, 76 * mm, 60 * mm]),
        Spacer(1, 8),
        callout(
            "历史对照支持这一判断",
            "同一中国数据的旧 primary-only 模型预处理后约 896 万变量、1.14 万约束。旧日志中首次非零 oneopt incumbent 在 655 秒出现，根 LP 到 7,595–7,617 秒仍未完成，并因当时的时间限制退出，gap 约 118,478%。本次 weighted 模型增加了构成变量和约 1.5 万行约束；截至当前，日志仍停在“transformed solutions”之后，说明瓶颈仍然是根 LP，只是更重。",
            color=AMBER,
            bg=colors.HexColor("#FFF8E8"),
        ),
        Spacer(1, 10),
        P("为什么没有再次出现 oneopt 非零解？", "CJKH2"),
        P(
            "旧 primary-only 目标只依赖 N 移动收益，oneopt 启发式较容易从单一线性收益结构构造一个非零整数解。当前目标同时包含源构成的 L1 辅助变量与环境收益，SCIP 转换了两个原始解（全零解及其辅助变量补全），但日志没有报告新的非零启发式解。可观测证据不足以断言具体启发式为何未触发；可以确定的是：根 LP尚未返回，因此没有 LP 解、reduced costs 或有效 dual bound供后续启发式使用。",
            "CJKBody",
        ),
        P("3. 冻结的数学模型", "CJKH1"),
        P("决策变量", "CJKH2"),
        P("对源区域 i、目的地区域 j、畜种 k，整数移动量为 x[i,j,k] ≥ 0。只有源端有该畜种且目的端允许接收时才建立变量。另设每个源区域的公共移出率 q[i]，以及每个源×畜种的绝对偏差 d[i,k]。", "CJKBody"),
        P("核心硬约束", "CJKH2"),
        bullet("源库存：Σ_j x[i,j,k] ≤ A[i,k]。"),
        bullet("源端氮安全：Σ_{j,k} n_out[i,k]·x[i,j,k] ≤ N_out[i] − 1 kg。"),
        bullet("目的地氮容量：Σ_{i,k} n_in[j,k]·x[i,j,k] ≤ −N_in[j]。"),
        bullet("目的地氨容量：Σ_{i,k} a_in[j,k]·x[i,j,k] ≤ NH3_in[j]。"),
        bullet("构成代理：d[i,k] ≥ x_out[i,k]/A[i,k] − q[i]，且 d[i,k] ≥ q[i] − x_out[i,k]/A[i,k]。"),
        PageBreak(),
    ]

    story += [
        P("4. 统一加权目标与归一化", "CJKH1"),
        callout(
            "统一目标",
            "maximize  J = 4·N̂ − 2·L̂ + Ê。三个分量都在求解前用固定物理量或解析上界归一化，避免 kg、比例和环境分的原始数量级直接竞争；整个目标再统一乘 1,000,000，仅改善数值尺度，不改变最优解。",
            color=TEAL,
            bg=LIGHT_TEAL,
        ),
        Spacer(1, 8),
        table([
            ["分量", "定义", "方向 / 范围", "中国固定标度"],
            ["N̂", "已消解源端氮 / 源端总氮", "最大化；[0,1]", "输入决定的源端总氮"],
            ["L̂", "等源权重、源内等畜种权重的移出率绝对偏差均值", "最小化；[0,1]；线性代理", "1,335 个源区域平均"],
            ["Ê", "目的地环境分 / 库存导出的理论上界", "最大化；[0,1]", "上界 35,410,051,488"],
            ["环境内部", "2·(1−敏感度) + 1·(1−PM2.5相关性)", "目的地系数", "内部权重 2:1"],
        ], [20 * mm, 73 * mm, 37 * mm, 40 * mm]),
        Spacer(1, 8),
        P("中国数值尺度", "CJKH2"),
        bullet("归一化后，单个移动变量的 N+环境绝对目标系数约为 1.33×10⁻¹⁰ 至 8.91×10⁻⁸。"),
        bullet("整体乘 10⁶ 后，求解器看到的对应系数约为 1.33×10⁻⁴ 至 8.91×10⁻²。该缩放不会改变 4:2:1，也不会改变可行域。"),
        bullet("当前模型故意不含移动成本、总移动上限或源区域最低保有率；这是为跨国复现 EU 原始 weighted 基线而冻结的设计，不应把它误解为政策完整模型。"),
        P("5. 证书规划：要证明什么", "CJKH1"),
        table([
            ["证书层级", "要求", "中国当前状态"],
            ["可行性证据", "至少一个满足整数和全部物理约束的解", "全零解满足；但科学价值弱"],
            ["非零 incumbent", "J_primal > 0 的可行转移方案", "尚未获得"],
            ["全局上界", "根 LP / 分支定界产生 J_dual", "尚未获得"],
            ["相对 gap", "SCIP native gap = |dual−primal| / min(|primal|,|dual|)", "primal=0 且 dual 缺失，当前不可定义"],
            ["成功终止", "optimal，或 gaplimit 且 native gap ≤ 0.001", "未达到"],
            ["结果完整性", "summary JSON + workbooks + 独立验证 all_passed", "求解终止前不会生成"],
        ], [29 * mm, 87 * mm, 54 * mm]),
        Spacer(1, 8),
        P("边界含义", "CJKH2"),
        P("这是最大化问题。incumbent/primal bound 是已知可行下界，dual bound 是全局上界。0.1% 证书只证明统一目标 J 距全局最优不超过 solver 定义的相对 gap；它不证明 N̂、L̂ 或 Ê 任一分量分别最优，也不证明政策意义上的运输成本合理。", "CJKBody"),
        PageBreak(),
    ]

    story += [
        P("6. 计算计划、时间估计与风险", "CJKH1"),
        P("已完成实例", "CJKH2"),
        table([
            ["实例", "总变量 / 约束", "证书", "SCIP 时间", "峰值 RSS"],
            ["EU", "71,774 / 2,019", "gaplimit 0.01008%", "15.79 s", "0.62 GiB"],
            ["澳大利亚", "389,583 / 6,210", "gaplimit 0.06635%", "3,104.21 s", "3.92 GiB"],
            ["中国", "10,663,406 / 26,811", "运行中；只有 J=0", ">43 h", "68.4 GiB"],
        ], [28 * mm, 42 * mm, 41 * mm, 28 * mm, 31 * mm]),
        Spacer(1, 8),
        callout(
            "不能给出可靠完成时间",
            "中国尚未完成第一次根 LP，因此没有 LP 迭代进度、dual bound、非零 incumbent 或 gap 轨迹。缺少这些量时，任何“还需几小时/几天”的数字都会是伪精确。唯一可靠的下界是：当前实现已经需要超过 43 小时，而且瓶颈仍未跨过。",
            color=RED,
            bg=colors.HexColor("#FFF1F0"),
        ),
        Spacer(1, 8),
        P("后续国家的粗规模估计（不是运行时预测）", "CJKH2"),
        table([
            ["实例", "整数移动变量", "相对澳大利亚", "按中国内存/变量粗估 RSS", "风险"],
            ["美国", "约 7.49M", "19.3×", "约 48 GiB", "根 LP 可能仍为小时至天级"],
            ["中国", "10.65M", "27.5×", "实测 68.4 GiB", "已超过 43 h，尚无根 LP结果"],
            ["巴西", "约 27.23M", "70.3×", "约 175 GiB", "建模/根 LP/序列化风险最高"],
        ], [25 * mm, 32 * mm, 30 * mm, 43 * mm, 40 * mm]),
        Spacer(1, 8),
        P("注：RSS 粗估只按中国当前的每总变量内存比例线性外推；真实值会受预处理删除率、约束非零结构、LP basis、cuts 和求解阶段影响。变量数不能可靠预测运行时间。", "CJKSmall"),
        P("7. 当前行动建议", "CJKH1"),
        table([
            ["方案", "科学含义", "建议"],
            ["继续当前运行", "保持 amendment 11 的稠密模型与 0.1% 证书完全不变；累积最干净的负面可扩展性证据", "当前默认；服务健康、内存充足"],
            ["增加 warm start / 构造非零解", "可能改善 primal，但不会直接解决根 LP上界；属于实现变更", "需先暂停并锁定新协议"],
            ["候选路线剪枝 / 列生成 / 分解", "改变可行域或求解算法，是最有希望的扩展路径", "作为下一轮研究，不可静默替换当前实验"],
            ["调整 LP/SCIP 参数或换原生建模", "可减少接口/根 LP成本，但会改变计算协议", "先保存本轮，再做受控对照"],
            ["取消 0.1% 证书只取可行解", "从“有界质量”变成启发式政策方案", "只有用户明确改变科学标准时采用"],
        ], [35 * mm, 86 * mm, 49 * mm]),
        Spacer(1, 9),
        callout(
            "本报告的判断边界",
            "我们可以高置信地说“中国仍停在根 LP，0 解只是天然 incumbent，尚无证书”；不能从当前证据断言根 LP最终一定完成、模型没有非零解、或预计某个具体日期完成。当前求解继续运行，未被本次诊断或 PDF 生成打断。",
            color=GREEN,
            bg=colors.HexColor("#ECF9F1"),
        ),
    ]

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
