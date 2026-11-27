"""Season free throws per player from saved basketball-reference totals pages: nba_ft_extract.py totals_2023.html totals_2024.html > docs/brier_site/nba_ft.csv
A traded player's season is his largest-FTA row (the multi-team total). Pages: https://www.basketball-reference.com/leagues/NBA_{year}_totals.html"""
import re, sys, csv
w = csv.writer(sys.stdout); w.writerow(["season", "player_id", "player", "ft", "fta"])
for path in sys.argv[1:]:
    season = re.search(r"(\d{4})\.html$", path).group(1); best = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", open(path, encoding="utf-8").read(), re.S):
        cell = dict(re.findall(r'data-stat="(\w+)"[^>]*>(.*?)</t[dh]>', row, re.S))
        m = re.search(r'data-append-csv="(\w+)"', row)
        if not m or "fta" not in cell: continue
        num = lambda k: int(re.sub(r"<.*?>", "", cell[k]) or 0)
        name = re.sub(r"<.*?>", "", cell["name_display"])
        if m.group(1) not in best or num("fta") > best[m.group(1)][2]:
            best[m.group(1)] = (name, num("ft"), num("fta"))
    for pid, (name, ft, fta) in sorted(best.items()): w.writerow([season, pid, name, ft, fta])
