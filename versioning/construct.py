import pandas as pd
import os 
from loguru import logger
from dotenv import load_dotenv
from delphi_epidata import Epidata
from config import SERIES, REGIONS, DELTA_DIR
from utils import check_missing_epiweek

load_dotenv()
api_key = os.getenv('DELPHI_API_KEY')
Epidata.auth = ("epidata", api_key) #type:ignore


API_ENDPOINT = {
    "wili" : Epidata.fluview
}

def pull_finalized_series(start_yw: int, 
                           end_yw: int, 
                           series: str = SERIES, 
                           regions: list[str] = REGIONS) ->pd.DataFrame:  #type:ignore
    
    api_fn = API_ENDPOINT[series]
    res = api_fn(
        regions,  # Fixed: use parameter instead of global REGIONS
        epiweeks=Epidata.range(start_yw, end_yw)
    )["epidata"]  # type: ignore
    return pd.DataFrame(res) 

def pull_finalized_info(df) -> tuple[dict[int,int], list[int]]: 
    max_issue_lookup = {item['epiweek']:item['issue'] 
                        for item in df.to_dict(orient="records")}
    epiweeks = df['epiweek'].sort_values().tolist()
    logger.info(f"Finalized series. Max epiweek: {max(epiweeks)}, min epiweek: {min(epiweeks)}")
    return max_issue_lookup, epiweeks

def build_base(df: pd.DataFrame, 
               initial_issue: int) -> pd.DataFrame: 
    is_finalized = df['issue'] <= initial_issue
    base = df[is_finalized]
    logger.info(f"Base series. Max epiweeek: {base['epiweek'].max()}, min epiweek: {base['epiweek'].min()}")
    return base

def calculate_delta(issue: int,
                    start_yw: int,
                    epiweeks: list[int], 
                    series: str = SERIES, 
                    regions: list[str] = REGIONS):
    api_fn = API_ENDPOINT[series]
    res = api_fn(
        regions,
        epiweeks=Epidata.range(start_yw, issue), 
        issues=issue
    )["epidata"]  # type: ignore

    df = pd.DataFrame(res)
    check_missing_epiweek(
        epiweeks=epiweeks,
        returned_epiweeks=df["epiweek"].tolist(),
        start_yw=start_yw,
        end_yw=issue
    )

    logger.info(f"Delta for issue={issue} has date range [{start_yw}, {issue}].")
    return pd.DataFrame(res)
        

def build_asof(target_issue:int, 
               delta_metadata:dict, 
               max_issue_lookup:dict, 
               base_series:pd.DataFrame) -> pd.DataFrame:
    
    def step_back_one_epiweek(current_yw: int) -> int:
        year, week = divmod(current_yw, 100)
        return (year - 1) * 100 + 52 if week == 1 else current_yw - 1

    result = []
    current_issue = target_issue
    current_start_yw = None
    
    # Get base series coverage
    base_max_epiweek = base_series['epiweek'].max()


    while (current_start_yw is None or current_start_yw > base_max_epiweek) and \
            str(current_issue) in delta_metadata:
            
            file_path = DELTA_DIR / f"asof={current_issue}.parquet"
            
            # Extra safety: stop if the file is missing
            if not file_path.exists():
                logger.warning(f"Expected delta file missing: {file_path}. Stopping traversal.")
                break

            delta = pd.read_parquet(file_path)
            
            if current_start_yw is not None:
                delta = delta[delta['epiweek'] < current_start_yw]
            
            result.append(delta)
            
            # Get metadata for next jump
            metadata = delta_metadata[str(current_issue)]
            current_start_yw = metadata["start_epiweek"]
            
            prev_yw = step_back_one_epiweek(current_start_yw)
            
            # JUMP: If the lookup doesn't exist, we must break
            current_issue = max_issue_lookup.get(str(prev_yw))
            if current_issue is None:
                break

    result.append(base_series)
    return pd.concat(result).sort_values('epiweek').reset_index(drop=True)
