from loguru import logger

def step_epiweeks(epiweeks: list, current_yw: int, k: int) -> int | None:
    try:
        current_idx = epiweeks.index(current_yw)
        target_idx = current_idx + k
        
        # Check both upper AND lower bounds
        if target_idx >= len(epiweeks) or target_idx < 0:
            return None
            
        return epiweeks[target_idx]
    except ValueError:
        logger.error(f"Epiweek {current_yw} not found in series.")
        return None

def earliest_active_epiweek(epiweeks: list[int], lookup: dict[int,int], issue: int, start_yw: int):
    i0 = epiweeks.index(start_yw)
    for t in epiweeks[i0:]:
        if lookup[t] > issue:
            logger.info(f"Issue {issue} is active as of epiweek={t} with max issue {lookup[t]} > {issue}.")
            return t
    return None

def check_missing_epiweek(epiweeks: list[int], 
                          returned_epiweeks: list[int],
                          start_yw: int, 
                          end_yw: int): 
    
    start_idx, end_idx = epiweeks.index(start_yw), epiweeks.index(end_yw)
    if set(returned_epiweeks) != set(epiweeks[start_idx: end_idx+1]):
        missing_weeks = list(set(epiweeks[start_idx: end_idx+1]) - set(returned_epiweeks))
        logger.info(f"Queried {start_yw}-{end_yw}, fixed to issue={end_yw} returned {min(returned_epiweeks)}-{max(returned_epiweeks)}.")
        if missing_weeks != []:
            logger.warning(f"Missing epiweeks detected from issue={end_yw} query: {missing_weeks}.")
        return missing_weeks
    return []