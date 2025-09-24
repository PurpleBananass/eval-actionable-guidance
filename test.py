# from pathlib import Path
# from hyparams import EXPERIMENTS, PROPOSED_CHANGES
# models = ["LightGBM","CatBoost"]
# ex = "SQAPlanner_confidence"
# missing = []
# for m in models:
#     for p in sorted(Path(EXPERIMENTS).glob("*@*")):
#         f = p / m / f"{ex}_all.csv"
#         if not f.exists():
#             missing.append(str(f))
# print("Missing flipped files:", len(missing))
# print("\n".join(missing[:15]), "...")
# print("Any plans?", any((Path(PROPOSED_CHANGES)/f"{p.name}/{m}/{ex}/plans_all.json").exists()
#                         for p in Path(EXPERIMENTS).glob("*@*")))
