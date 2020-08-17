import os
import sys

import numpy as np
import pandas as pd



from optparse import OptionParser
from multiprocessing import Pool
from pathlib import Path

from june.logger.read_logger import ReadLogger
from june.infection import SymptomTag



def save_world_summaries(logger,summary_path,daily_summary_path):
    world_df = logger.world_summary()
    world_df.to_csv(summary_path)

    world_df['new_hospital_dead'] = logger.get_daily_updates(logger.infections_df, 
        lambda x: (
            x.infected_id[(x.symptoms == SymptomTag.dead_icu) | (x.symptoms == SymptomTag.dead_hospital)]
        )
    )

    total_pop_cols = ['infected', 'recovered', 'dead', 'susceptible']
    world_pop = world_df[total_pop_cols].iloc[0].sum()

    world_df['seroprevalence'] = 100.*world_df['new_infections'].cumsum()/world_pop

    keep_cols = [
        'new_hospital_dead',
        'new_infections',
        'hospital_admissions',
        'intensive_care_admissions'
    ]

    daily_world_df = world_df[keep_cols].resample('D').sum()
    daily_world_df['seroprevalence'] = (
        100.*daily_world_df['new_infections'].cumsum()/world_pop
    )

    daily_world_df.to_csv(daily_summary_path)   
    return None

def save_age_summaries(logger, age_summary_path, daily_age_summary_path, age_bins=None):
    '''some of the pandas commands are suspect...'''

    if age_bins is None:
        age_bins = [0, 6, 18, 65, 85, 100]

    age_df = logger.age_summary(age_bins)
    age_df.index = pd.to_datetime(age_df.index)  
    age_df.set_index(['age_range',age_df.index],inplace=True)

    total_pop_cols = ['infected', 'recovered', 'dead', 'susceptible']
    age_pops = age_df.groupby('age_range')[total_pop_cols].nth(0).sum(axis=1)

    age_df['new_hospital_dead'] = pd.Series(index=age_df.index)
    age_df['seroprevalence'] = pd.Series(index=age_df.index)
    # set as empty series, with same index.

    for jj,(age_range,gdf) in enumerate(age_df.groupby('age_range')):
        # gdf is a multi-index, with Index level0='0_5', level1=[list of timestamps]
        # can use subgroup index as "complete indexing" to set on larger df.

        sero = 100.*gdf['new_infections'].cumsum()/age_pops.iloc[jj]
        age_df.loc[gdf.index,'seroprevalence'] = sero

        min_age = int(age_range.split('_')[0])
        max_age = int(age_range.split('_')[1])

        nhd = logger.get_daily_updates(logger.infections_df, 
            lambda x: (
                x.infected_id[
                    ((x.symptoms == SymptomTag.dead_icu) | (x.symptoms == SymptomTag.dead_hospital))
                    & ((min_age <= x.age) & (x.age <= max_age))
                ]
            )
        )

        age_df.loc[gdf.index,'new_hospital_dead'] = nhd.values.astype(int)
        # end of "for" loop - noted bc. I was confused by all the indention
        

    # undo the MultiIndex...
    age_df.reset_index(level=0,inplace=True)
    age_df.to_csv(age_summary_path)

    # now do the resampled version!
    keep_cols = [
        'new_hospital_dead',
        'new_infections',
        'hospital_admissions',
        'intensive_care_admissions',
        'age_range'
    ]

    # can't just copy seroprevalence, as it's a cumulative.    
    daily_age_df = age_df[keep_cols].groupby('age_range').resample('D').sum()
    daily_age_df['seroprevalence'] = pd.Series(index=daily_age_df.index)
    # set as empty series, with same index.

    # no need to recalc. age_pops

    for jj,(age_range,gdf) in enumerate(daily_age_df.groupby('age_range')):
        # gdf is a multi-index, with Index level0='0_5', level1=[list of timestamps]
        # can use subgroup index as "complete indexing" to set on larger df.
        sero = 100.*gdf['new_infections'].cumsum()/age_pops.iloc[jj]
        daily_age_df.loc[gdf.index,'seroprevalence'] = sero

    # undo the MultiIndex...
    daily_age_df.reset_index(level=0,inplace=True)
    daily_age_df.to_csv(daily_age_summary_path)

    return None

def save_hospital_summary(logger,hospital_summary_path):
    hospital_df = logger.load_hospital_capacity()
    hospital_df.to_csv(hospital_summary_path)

    return None

def save_infection_locations(logger,infection_locations_path):
    inf_loc_df = logger.get_locations_infections()
    inf_loc_df.to_csv(infection_locations_path)

    return None

def save_location_infections_timeseries(
    logger,inf_loc_timeseries_path, daily_inf_loc_timeseries_path
):
    inf_loc_ts = logger.get_location_infections_timeseries()
    inf_loc_ts.to_csv(inf_loc_timeseries_path)

    inf_loc_ts.index = pd.to_datetime(inf_loc_ts.index)
    daily_inf_loc_ts = inf_loc_ts.resample('D').sum()
    daily_inf_loc_ts.to_csv(daily_inf_loc_timeseries_path)
    return None


if __name__ == "__main__":  
    print('no main block')











