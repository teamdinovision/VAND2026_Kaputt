import pandas as pd

prediction = pd.read_csv("fused_rank_avg.csv")
small_image = pd.read_csv("small_image_k2.csv")

small_ids = set(small_image["capture_id"])
prediction.loc[prediction["capture_id"].isin(small_ids), "pred"] = 0.01

prediction.to_csv("prediction_d_replaced.csv", index=False)
print(f"Done. Replaced {prediction['capture_id'].isin(small_ids).sum()} rows. Output: prediction_d_replaced.csv")
