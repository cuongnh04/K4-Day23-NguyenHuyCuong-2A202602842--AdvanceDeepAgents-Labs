"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    by_number = {}
    by_url = set()
    for source in sources:
        number = source.get("n") if isinstance(source, dict) else None
        url = source.get("url") if isinstance(source, dict) else None
        if not isinstance(number, int):
            problems.append(f"source n={number!r} is not an int")
        elif number in by_number:
            problems.append(f"source [{number}] appears more than once")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source [{number}] url is not http(s)")
        elif url in by_url:
            problems.append(f"duplicate source URL: {url}")
        else:
            by_url.add(url)

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if not heading:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    citation_pattern = re.compile(r"\[(\d+)\]")
    cited = {int(number) for number in citation_pattern.findall(body)}
    for number in sorted(cited - set(by_number)):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(set(by_number) - cited):
        problems.append(f"source [{number}] never cited")

    ref_lines = [line for line in references.splitlines() if re.match(r"^\s*\[\d+\]", line)]
    seen_refs = {}
    for line in ref_lines:
        match = re.match(r"^\s*\[(\d+)\]\s+", line)
        if not match:
            continue
        number = int(match.group(1))
        urls = re.findall(r"https?://\S+", line)
        if number in seen_refs:
            problems.append(f"reference [{number}] appears more than once")
        seen_refs[number] = line
        if number not in by_number:
            problems.append(f"reference [{number}] is not in sources.json")
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and urls[0].rstrip(".,)") != by_number[number]["url"]:
            problems.append(f"reference [{number}] URL does not match sources.json")
    for number in sorted(set(by_number) - set(seen_refs)):
        problems.append(f"missing reference line for source [{number}]")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
