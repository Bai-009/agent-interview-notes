#!/usr/bin/env python3
"""仓库自检 —— 只查机械一致性，不评内容。

动机（2026-08-29）：一次全仓审查发现的缺陷里，绝大多数是**读文件读不出来**的——
断链、题号错配、计数漂移、图与资产表不同步、表格被空行截断。
这类问题靠人工复读发现不了，靠脚本一秒发现。所以固化成这个文件。

用法：  python3 _工具/自检.py
退出码：0 = 全通过，1 = 有问题。

⚠️ 这不是 SOP 的强制步骤。小轮次不必跑；归档完一题、动过索引结构、
或准备推公开仓之前跑一下即可。检查项该增就增——它和规范一样是活的。
"""

import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

RAW_DIR = "_原始记录"          # 一手材料，除断链外一律不检查
problems: list[str] = []


def md_files(skip_raw: bool = True):
    for p in sorted(glob.glob("**/*.md", recursive=True)):
        if ".git" in p:
            continue
        if skip_raw and p.startswith(RAW_DIR):
            continue
        yield p


def read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def line_of(text: str, idx: int) -> int:
    return text[:idx].count("\n") + 1


def question_files() -> list[str]:
    return sorted(glob.glob("0[1-6]-*/Q*.md"))


def is_stub(path: str) -> bool:
    """题目档：题目已给出但未作答（掌握度 🔄进行中）。只有题面与空白作答区，
    不该按完成记录的标准去查必备节，也不计入完成数。"""
    return "掌握度: 🔄进行中" in read(path)


def report(name: str, bad: list[str], detail_limit: int = 20):
    if bad:
        problems.extend(bad)
        print(f"❌ {name}：{len(bad)} 处")
        for b in bad[:detail_limit]:
            print(f"     {b}")
        if len(bad) > detail_limit:
            print(f"     …… 另有 {len(bad) - detail_limit} 处")
    else:
        print(f"✅ {name}")


# ── 1. 相对链接断链 ────────────────────────────────────────────────
def check_relative_links():
    import urllib.parse
    bad = []
    for p in md_files(skip_raw=False):
        text = read(p)
        for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = urllib.parse.unquote(target.split("#")[0])
            if not path:
                continue
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(p), path))):
                bad.append(f"{p}:{line_of(text, m.start())} -> {target}")
    report("相对链接无断链", bad)


# ── 2. 双链能否解析（Obsidian 口径）────────────────────────────────
def planned_names() -> set[str]:
    """从 题库进度.md 的「规划文件名」表读未作答题的目标名，避免两处硬编码。"""
    text = read("00-索引/题库进度.md")
    section = text.split("## 🔗 规划文件名")
    if len(section) < 2:
        return set()
    body = section[1].split("\n---")[0]
    return set(re.findall(r"`(Q\d{2}-[^`]+)`", body))


def check_wikilinks():
    existing = {os.path.splitext(os.path.basename(p))[0] for p in md_files(skip_raw=False)}
    planned = planned_names()
    bad, resolved, pending = [], 0, 0
    for p in md_files():
        text = read(p)
        stripped = re.sub(r"`[^`]*`", "", text)   # 反引号里的是示意写法，不算链接
        for m in re.finditer(r"\[\[([^\]]+)\]\]", stripped):
            # 表格里的别名双链必须写成 [[名字\|别名]]，解析时把转义的反斜杠一并去掉
            name = re.split(r"\\?\|", m.group(1))[0].strip()
            if name in existing:
                resolved += 1
            elif name in planned:
                pending += 1
            else:
                bad.append(f"{p}:{line_of(stripped, m.start())} [[{name}]]"
                           f"  ← 既非现有文件，也不在规划文件名表里")
    report(f"双链可解析（现有 {resolved} / 待建 {pending}）", bad)


def check_table_pipes():
    """表格单元格里的 [[名字|别名]] 会在竖线处被切成两格——别名和表格一起坏。
    GFM 与 Obsidian 都要求写成 [[名字\\|别名]]。"""
    bad = []
    for p in md_files():
        for i, line in enumerate(read(p).split("\n"), 1):
            if line.lstrip().startswith("|") and re.search(r"\[\[[^\]\\|]*?\|", line):
                bad.append(f"{p}:{i} 表格内的别名双链竖线未转义，应写 [[名字\\|别名]]")
    report("表格内别名双链已转义", bad)


# ── 3. 表格被空行截断（GitHub 上会渲染成纯文本）─────────────────────
def check_tables():
    bad = []
    for p in md_files():
        lines = read(p).split("\n")
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("|"):
                start, block = i, []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                sep_ok = len(block) >= 2 and set(block[1].replace("|", "").replace(" ", "")) <= set("-:")
                if not sep_ok:
                    bad.append(f"{p}:{start + 1} 表块缺表头分隔行（{len(block)} 行）")
            else:
                i += 1
    report("表格结构完整", bad)


# ── 4. 用词规范（词表直接读 CLAUDE.md 的对照表）─────────────────────
def banned_terms() -> set[str]:
    text = read("CLAUDE.md")
    sec = re.search(r"## 5\.5.*?(?=\n## )", text, re.S)
    if not sec:
        return set()
    terms = set()
    for row in re.finditer(r"^\|\s*([^|]+?)\s*\|\s*[^|]+\|\s*$", sec.group(0), re.M):
        left = row.group(1)
        if left in ("不用", "---") or set(left) <= set("-: "):
            continue
        for t in re.sub(r"（[^）]*）", "", left).split("/"):
            t = t.strip().strip("*")
            if t:
                terms.add(t)
    return terms


def strip_primary_quotes(text: str) -> str:
    """第 2 / 2.5 节的引文是一手材料，用词规范对它不适用（CLAUDE.md 5.5 的例外条款）。
    置空而不删行，保持行号不变。"""
    out, in_primary = [], False
    for line in text.split("\n"):
        if line.startswith("## "):
            in_primary = bool(re.match(r"## 2(\.5)?[ .]", line))
        out.append("" if in_primary and line.lstrip().startswith(">") else line)
    return "\n".join(out)


def check_wording():
    terms = banned_terms()
    if not terms:
        report("用词规范", ["无法从 CLAUDE.md 5.5 解析出词表"])
        return
    bad = []
    for p in md_files():
        text = strip_primary_quotes(read(p))
        if p == "CLAUDE.md":                       # 对照表本身与其上下文豁免
            text = re.sub(r"## 5\.5.*?(?=\n## )", "", text, flags=re.S)
        for n, line in enumerate(text.split("\n"), 1):
            for t in terms:
                if t in line:
                    bad.append(f"{p}:{n} 含「{t}」")
    report(f"用词规范（{len(terms)} 个禁用词）", bad)


# ── 5. 知识网络 ↔ 公共资产表 边一致性 ──────────────────────────────
def check_graph_edges():
    # 实线边只画在**已有记录**的题之间；指向未作答题（含题目档）的是虚线，不参与本项比对
    done = {os.path.basename(p)[:3] for p in question_files() if not is_stub(p)}
    prog = read("00-索引/题库进度.md")
    rows = [l for l in prog.split("\n") if l.startswith("| **") and l.count("|") >= 4]
    asset = set()
    for l in rows:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) < 3:
            continue
        src = {q for q in re.findall(r"Q\d{2}", cells[1]) if q in done}
        dst = {q for q in re.findall(r"Q\d{2}", cells[2]) if q in done}
        for a in src:
            for b in src:
                if a < b:
                    asset.add((a, b))
            for b in dst:
                if a != b:
                    asset.add(tuple(sorted((a, b))))
    net = set()
    for m in re.finditer(r"(Q\d{2})\s*-->\|[^|]*\|\s*(Q\d{2})", read("00-索引/知识网络.md")):
        if m.group(1) in done and m.group(2) in done:
            net.add(tuple(sorted((m.group(1), m.group(2)))))
    bad = [f"资产表有、图上没画：{a}–{b}" for a, b in sorted(asset - net)]
    bad += [f"图上有、资产表没登记：{a}–{b}" for a, b in sorted(net - asset)]
    report(f"知识网络与资产表同步（各 {len(net)} 条边）", bad)


# ── 6. 记录必备节 ─────────────────────────────────────────────────
def check_sections():
    need = [(r"^## ⚡ 30 秒速览", "速览"),
            (r"^## 2\. ", "原始回答"),
            (r"^## 7\.5 ", "查证记录"),
            (r"^## 9\. ", "核心结论"),
            (r"^## 10\. ", "复习记录")]
    bad = []
    for p in question_files():
        if is_stub(p):                             # 题目档只有题面，不适用
            continue
        text = read(p)
        for pat, label in need:
            if not re.search(pat, text, re.M):
                bad.append(f"{p} 缺「{label}」")
    report("记录必备节齐全", bad)


# ── 7. 计数一致性 ─────────────────────────────────────────────────
def check_counts():
    files = question_files()
    done = [p for p in files if not is_stub(p)]
    stubs = [p for p in files if is_stub(p)]
    bad = []
    for p, pat, n, label in [("README.md", r"`进度 (\d+) / 22`", len(done), "记录"),
                             ("README.md", r"`进行中 (\d+)`", len(stubs), "题目档"),
                             ("00-索引/题库进度.md", r"`已完成 (\d+) / 22`", len(done), "记录"),
                             ("00-索引/题库进度.md", r"`进行中 (\d+)`", len(stubs), "题目档")]:
        m = re.search(pat, read(p))
        if not m:
            bad.append(f"{p} 找不到徽章 {pat}")
        elif int(m.group(1)) != n:
            bad.append(f"{p} 写 {m.group(1)}，实际有 {n} 篇{label}")
    m = re.search(r"`已基线核查 (\d+)`", read("00-索引/题库进度.md"))
    n76 = sum(1 for p in done if re.search(r"^## 7\.6 ", read(p), re.M))
    if m and int(m.group(1)) != n76:
        bad.append(f"题库进度写「已基线核查 {m.group(1)}」，实际有 7.6 节的是 {n76} 篇")
    report(f"计数一致（{len(done)} 篇记录 / {len(stubs)} 篇题目档）", bad)


# ── 8. 索引徽章日期跟上修改 ─────────────────────────────────────────
def check_badge_dates():
    """索引页顶部的「更新: YYYY-MM-DD」徽章是读者判断新旧的唯一依据。
    约定：徽章 = 该文件最后一次被编辑的日期——干净文件对齐它最后一次提交，
    工作区里改过的文件必须写当天。2026-09-02 加：当时 6 个索引文件的徽章落后于实际修改。"""
    import subprocess, datetime
    today = datetime.date.today().isoformat()

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    bad = []
    for p in ["README.md", "07-补充议题/README.md"] + sorted(glob.glob("00-索引/*.md")):
        m = re.search(r"`更新:? ?(\d{4}-\d{2}-\d{2})", read(p))
        if not m:
            continue
        dirty = git("status", "--porcelain", "--", p) != ""
        last = git("log", "-1", "--format=%ad", "--date=short", "--", p)
        want = today if dirty else last
        if want and m.group(1) != want:
            why = "工作区已修改，应写当天" if dirty else f"最后提交于 {last}"
            bad.append(f"{p} 徽章写 {m.group(1)}，应为 {want}（{why}）")
    report("索引徽章日期跟上修改", bad)


# ── 9. 量词巡检（只列不判）────────────────────────────────────────────
QUANTIFIERS = r"主流|通常|大多数|绝大多数|一般来说|普遍|几乎所有"


def list_quantifiers(verbose: bool):
    """台账「反例巡检」观察项的机械部分：把带量词的技术论断列出来，判断留给人。
    不计入失败。跳过一手材料（_原始记录、第 2/2.5 节引文）、模板与规范本身，
    以及引号里讨论量词本身的句子。2026-09-02 首次运行即查出 Q09 两处改漏的「同步仍是主流」。"""
    hits = []
    for p in md_files():
        if p.startswith("_模板/") or p == "CLAUDE.md":
            continue
        text = strip_primary_quotes(read(p))
        for n, line in enumerate(text.split("\n"), 1):
            if re.search(QUANTIFIERS, line) and not re.search(r"量词|反例巡检|「[^」]*主流[^」]*」", line):
                hits.append(f"{p}:{n} {line.strip()[:90]}")
    tail = "" if verbose else "；加 --巡检 看明细"
    print(f"ℹ️ 量词巡检：{len(hits)} 处（只列不判{tail}）")
    if verbose:
        for h in hits:
            print(f"     {h}")


if __name__ == "__main__":
    print(f"仓库自检 · {ROOT}\n" + "─" * 60)
    for fn in (check_relative_links, check_wikilinks, check_table_pipes, check_tables, check_wording,
               check_graph_edges, check_sections, check_counts, check_badge_dates):
        fn()
    list_quantifiers(verbose="--巡检" in sys.argv)
    print("─" * 60)
    if problems:
        print(f"合计 {len(problems)} 处待处理")
        sys.exit(1)
    print("全部通过")
