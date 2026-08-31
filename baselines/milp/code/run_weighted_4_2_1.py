import json
import os
import sys
from pathlib import Path
from time import perf_counter

from dataLoader import load_datas
from weighted_4_2_1 import solve_weighted_4_2_1


FILES = {
    "cn": "中国国家尺度更新PB第一步.xlsx",
    "eu": "欧盟更新PB后第一步.xlsx",
    "aus": "澳大利亚空间优化更新PB第一步.xlsx",
    "usa": "美国数据国家尺度第一步1223.xlsx",
    "br": "巴西指标国家优化更新PB第一步.xlsx",
}


def main():
    country = sys.argv[1] if len(sys.argv) > 1 else "eu"
    if country not in FILES:
        raise SystemExit(f"unsupported weighted dataset: {country}")
    preflight_only = "--preflight" in sys.argv[2:]
    input_started = perf_counter()
    data = load_datas(country, FILES[country])
    input_elapsed = perf_counter() - input_started
    summary = solve_weighted_4_2_1(
        country,
        FILES[country],
        data,
        input_load_elapsed_seconds=input_elapsed,
        preflight_only=preflight_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
