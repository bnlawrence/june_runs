import os
import sys

import numpy as np
import pandas as pd



from optparse import OptionParser
from multiprocessing import Pool
from pathlib import Path

from june.logger.read_logger import ReadLogger
from june.infection import SymptomTag

def group_all_region_df(regional_run_summary, all_regions):
    regions_to_group = [['East Midlands', 'West Midlands'],
                        ['North East', 'Yorkshire and The Humber']]
    
    grouped = False
    grouped_dfs = {}
    for regions in regions_to_group:
        if set(regions).issubset(all_regions):
            grouped = True
            grouped_name = {'East Midlands_West Midlands': 'Midlands',
            'North East_Yorkshire and The Humber': 'North East and Yorkshire'}
            grouped_dfs[grouped_name['_'.join(regions)]] = group_region_df(regional_run_summary, regions)
            regional_run_summary.drop(regions, level=0, inplace=True)
    if grouped:
        grouped_dfs = pd.concat(grouped_dfs)
        regional_run_summary = pd.concat([regional_run_summary, grouped_dfs])
        regional_run_summary = regional_run_summary.dropna()
        regional_run_summary = regional_run_summary.astype(int)
        return regional_run_summary
    else:
        return regional_run_summary

def group_region_df(regional_run_summary, region_list):
    grouped_df = regional_run_summary.loc[region_list].sum(level=1).sort_index()
    return grouped_df

def save_regional_summaries(logger,run_summary_path,daily_regional_path):
    run_summary = logger.run_summary()
    run_summary.to_csv(run_summary_path)
    regional_run_summary = run_summary.groupby(['region', run_summary.index]).sum()
    all_regions = regional_run_summary.index.get_level_values(0).unique()

    total_pop_cols = ['current_infected',
                      'current_recovered',
                      'current_dead',
                      'current_susceptible']

    # group Midlands and North East and Yorkshire if they exist
    regional_run_summary = group_all_region_df(regional_run_summary, all_regions)

    regions = regional_run_summary.index.get_level_values(0).unique()
    time_stamps = regional_run_summary.index.get_level_values(1).unique()

    region_populations = {region: regional_run_summary[total_pop_cols].loc[region].iloc[0].sum() for region in regions}

    for region in regions:
        seroprev_df = 100. * regional_run_summary.loc[region, 'daily_infections'].cumsum() / region_populations[region]
        seroprev_df.index = pd.MultiIndex.from_product([[region], regional_run_summary.loc[region].index.get_level_values(0)])
        regional_run_summary.loc[region, 'seroprevalence'] = seroprev_df
    regional_run_summary.to_csv(daily_regional_path)
    return None

def save_world_summaries(logger,world_summary_path,daily_summary_path):
    world_df = logger.world_summary()
    world_df.to_csv(world_summary_path)

    total_pop_cols = ['current_infected',
                      'current_recovered',
                      'current_dead',
                      'current_susceptible']

    world_pop = world_df[total_pop_cols].iloc[0].sum()

    world_df['seroprevalence'] = 100.*world_df['daily_infections'].cumsum() / world_pop

    daily_world_df = world_df.filter(regex="daily_*").resample('D').sum()
    current_world_df = world_df.filter(regex="current_*").resample('D').last()
    daily_world_df = pd.concat([current_world_df, daily_world_df], axis=1)

    daily_world_df['seroprevalence'] = (
        100.*daily_world_df['daily_infections'].cumsum() / world_pop
    )

    daily_world_df.to_csv(daily_summary_path)   
    return None

def save_age_summaries(logger, age_summary_path, daily_age_summary_path, age_bins=None):
    '''some of the pandas commands are suspect...'''

    if age_bins is None:
        age_bins = [0, 6, 18, 65, 85, 100]
    elif age_bins == "individual":
        age_bins = np.arange(0,101)

    age_df = logger.age_summary(age_bins)
    age_df.to_csv(age_summary_path)

    total_pop_cols = ['current_infected',
                      'current_recovered',
                      'current_dead',
                      'current_susceptible']
    age_pops = age_df.groupby('age_range')[total_pop_cols].nth(0).sum(axis=1)

    dfs = []
    for age, group in age_df.groupby('age_range'):
        tmp_daily_df = group.filter(regex="daily_*").resample('D').sum()
        tmp_current_df = group.filter(regex="current_*").resample('D').last()
        tmp_daily_df = pd.concat([tmp_current_df, tmp_daily_df], axis=1)
        tmp_daily_df['seroprevalence'] = 100. * tmp_daily_df['daily_infections'].cumsum() / age_pops[age]
        tmp_daily_df.insert(0, 'age_range', age)
        dfs.append(tmp_daily_df)
    daily_age_df = pd.concat(dfs)        
    daily_age_df.to_csv(daily_age_summary_path)

    return None

def save_hospital_summary(logger,hospital_summary_path):
    hospital_df = logger.load_hospital_capacity()
    hospital_df.to_csv(hospital_summary_path)

    return None

def save_infection_locations(logger,infection_locations_path,daily_inf_loc_timeseries_path):
    infection_locations = logger.get_locations_infections()
    unique_locations, counts_locations = np.unique(
                np.array(infection_locations),
                return_counts=True)

    location_counts_df = pd.DataFrame(
        {'locations': unique_locations,
        'counts': counts_locations}
    )
    location_counts_df.set_index('locations', inplace=True)
    location_counts_df['percent_infections']= 100*(location_counts_df['counts'])/location_counts_df.values.sum()
    location_counts_df.to_csv(infection_locations_path)

    inf_loc_daily_ts = pd.DataFrame()
    for location in unique_locations:
        inf_loc_daily_ts[str(location)] = logger.locations_df.apply(
        lambda x: x.location.count(str(location)), axis=1
        )
    inf_loc_daily_ts.index = pd.to_datetime(inf_loc_daily_ts.index)
    inf_loc_daily_ts.to_csv(daily_inf_loc_timeseries_path)

    return None

# def save_location_infections_timeseries(
#     logger,inf_loc_timeseries_path, daily_inf_loc_timeseries_path
# ):
#     inf_loc_ts = logger.get_location_infections_timeseries()
#     inf_loc_ts.to_csv(inf_loc_timeseries_path)

#     inf_loc_ts.index = pd.to_datetime(inf_loc_ts.index)
#     daily_inf_loc_ts = inf_loc_ts.resample('D').sum()
#     daily_inf_loc_ts.to_csv(daily_inf_loc_timeseries_path)
#     return None


if __name__ == "__main__":  
    print('no main block')











