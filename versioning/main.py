import time
import json
from loguru import logger
from utils import step_epiweeks, earliest_active_epiweek
from config import START_EPIWEEK, END_EPIWEEK, INIT_TRAIN, DATA_DIR, DELTA_DIR, API_SLEEP
from construct import pull_finalized_series, pull_finalized_info, build_base, calculate_delta

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(DATA_DIR / "epidata_versioning.log", mode="w")
    
    finalized_df = pull_finalized_series(start_yw=START_EPIWEEK, 
                                         end_yw=END_EPIWEEK)
    finalized_df.to_parquet(DATA_DIR / f"finalized.parquet")
    MAX_LOOKUP_TABLE, EPIWEEKS = pull_finalized_info(finalized_df)
    with open(DATA_DIR / "max_issue_lookup.json", "w") as f:
        json.dump(MAX_LOOKUP_TABLE, f, indent=4)

    
    init_issue = step_epiweeks(EPIWEEKS, START_EPIWEEK, INIT_TRAIN)
    base_series = build_base(finalized_df, init_issue) #type:ignore
    base_series.to_parquet(DATA_DIR / f"base.parquet")
    
    # start right after base
    start_yw = step_epiweeks(EPIWEEKS, int(base_series['epiweek'].max()), 1)
    logger.info(f"Initial start_yw {start_yw}")
    
    delta_metadata = {}
    for w in EPIWEEKS[EPIWEEKS.index(init_issue):]: #type:ignore
        delta = calculate_delta(issue=w,
                              start_yw=start_yw, #type:ignore
                              epiweeks=EPIWEEKS)
        
        file_path = DELTA_DIR / f"asof={w}.parquet"
        delta.to_parquet(file_path)

        delta_metadata[int(w)] = {
            "start_epiweek": int(start_yw), #type:ignore
            "end_epiweek": int(w),
            "filename": file_path.name,
            "rows": len(delta)
        }

        time.sleep(API_SLEEP)
        
        start_yw = earliest_active_epiweek(EPIWEEKS, MAX_LOOKUP_TABLE, 
                                          w, start_yw) #type:ignore
        if start_yw is None:
            logger.warning(f"issue={w} leads to start_yw={start_yw}")
            break

    with open(DATA_DIR / "delta_metadata.json", "w") as f:
        json.dump(delta_metadata, f, indent=4)
    
    logger.info("Metadata export complete.")


if __name__ == "__main__":
    main()

    



    
