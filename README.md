# ts-asof

The goal of this library is to help reconstruct versioned `as_of` time series snapshots. A necessary step for backtesting in nowcast environments. To be specific, given a finalized dataset, and access to revisions, this library reconstructs an `as_of` series for any time in the query window to allow users to obtain a snapshot of the series that was *as of* the desired date.

Notably, this setup assumes that:
1. You have access to a ‘finalized’ ground truth dataset, which reflects revisions over all time periods before the end date. 
2. You have access to ‘snapshots’ or ‘issues’ which are incomplete, but provide revisions of past data that were known as of the date of issue.

For example, in the `fluview` endpoint in the `epidata` api, there is an `issue` parameter that yeilds all revisions as of the issue date. So if I am interested in knowing what was known about 2019-01 *influenza-like-illness* statstics as of 2020-01, we can track all the issues between the initial release on 2019-01 until 2020-01. Ths doesn't prevent further updates or revisions of the 2019-01 records to be updated after 2020-01 of course, but it provides an accurate snapshot of 2019-01 statistics *as of* 2020-01. 

## Install
With `uv`:
```bash
uv add asof
```

and with `pip`, 
```bash 
pip install asof
```

A note on dependencies. The current iteration requires `epidata` as it was designed with the sole purpose of creating an `asof` version of the `fluview` endpoint. Future iterations can remove this dependencies or modify to opt into it. 

## Usage

Please see the */examples/wili_example.ipynb* for a demo and extra notes on usage. 

## Algorithm
**Step 1: Build Phase**
1. **Fetch finalized snapshot once**: Query all data with complete revision history `[start, end]`
2. **Identify stable base**: Find all time points finalized before an initial version threshold (these never change again)
3. **Build delta chain**: For each subsequent version *v*:
   - Fetch only the **active revision window** — time points where `max_version > v`
   - Skip time points already finalized (key optimization!)
   - Store sparse delta containing only revised values

**Step 2: Reconstruction**
To reconstruct the series as of version *V*:
1. Load delta file for version *V*
2. Extract its start time point *t*<sub>start</sub> from metadata
3. Step back one time point: *t*<sub>prev</sub> = *t*<sub>start</sub> - 1
4. Look up which version last modified *t*<sub>prev</sub> in the pre-computed `max_version_lookup`
5. Jump to that version's delta file and repeat
6. Terminate when reaching the frozen base
7. Concatenate: `base ⊕ delta_chain`

## Future Improvements 
1. Added flexibility. The main pain point in the current iteration is the lack of a unified `datetime` handling of the date indices. This is very fragile and would be the key item to address for future iterations. 
2. Allowing for a `step` parameter for reconstruction. This saves time and space if the user does not need `asof` reconstruction for every time step between `start_date + init_window_size` and `end_date`, but rather, every `step`. 
3. Support for other data streams. As of now, only `epidata` support for the `fluview` endpoint is available. This framework can easily be extended to other `epidata` API endpoints that don't have `asof` available. However, support for other streams (e.g., FRED) would require the user to build a custom `DataSource` abstract base class that mimics the implementation in `epidata.py` for `DelphiEpidata`.

