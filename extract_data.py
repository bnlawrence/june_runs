import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import shutil
import psutil

from june.logger.read_logger import ReadLogger
from june.infection import SymptomTag

import generate_parameters as gp

from optparse import OptionParser
from multiprocessing import Pool

import time



age_bins = [0,20,40,60,80,100]
strat_age_bins = [0, 6, 18, 65, 85, 100]
fine_age_bins = [0,10,20,30,40,50,60,70,80,90,100]

num_samples = 250

names = gp.all_names

lhs_array = gp.generate_lhs(num_samples=num_samples)

work_dir = f'/cosma7/data/dp004/dc-quer1/june_runs'

# Sometimes all of the n_threads catch on this point at once,
# so it tries to make dir that already exists - not good.
# So wrap it in try-except block!

def print_memory_status(when='now'):
    mem = psutil.virtual_memory()
    tot = f"total: {mem.total/1024**3:.2f}G"
    used = f"used: {mem.used/1024**3:.2f}G"
    perc = f"perc: {mem.percent:.2f}%"
    avail = f"avail: {mem.available/1024**3:.2f}G"
    print(f"mem {when}: {tot} {used} {perc} {avail}")



def do_extract(
    index,
    logger_dir,
    summary_dir=None,
    iteration=1):
        
    if os.path.exists(logger_dir) is False:
        raise IOError(f'No logger dir {logger_dir}!')

    if summary_dir is None:
        logger_end = [i for i in logger_dir.split('/') if len(i) > 0][-1]
        summary_dir = f'{work_dir}/june_results/summary/{logger_end}'

    if os.path.exists(summary_dir) is False:
        try:
            os.makedirs(summary_dir)
            print('made summary_dir')
        except:
            print('dir {summary_dir} already exists?')

    print(f'save at {summary_dir}')

    try:
        index = int(index)
        index = f'{index:03d}'
    except:
        pass
    

    inf_loc_path = f'{summary_dir}/infections_locations_{index}.csv'
    world_summary_path = f'{summary_dir}/world_summary_{index}.csv'
    daily_world_summary_path = f'{summary_dir}/daily_world_summary_{index}.csv'
    #age_summary_path = f'{summary_dir}/age_summary_{ii}.csv'
    strat_age_summary_path = f'{summary_dir}/strat_age_summary_{index}.csv'
    daily_strat_age_summary_path = f'{summary_dir}/daily_strat_age_summary_{index}.csv'

    #fine_age_summary_path = f'{summary_dir}/fine_age_summary_{ii}.csv'
    hospital_df_path = f'{summary_dir}/hospital_summary_{index}.csv'


    print('logger exists?:',os.path.exists(logger_dir+'/logger.hdf5'))
    print(logger_dir)

    try:
        t1 = time.time()
        logger = ReadLogger(logger_dir)
        t2 = time.time()

        print(f'logger {logger_dir} \n    read in {(t2-t1)/60.} min')
    except:
        err_str = ('****                        ****\n\n'
                + f'    Cannot read logger {index}!    \n\n'
                +  '****                        ****\n')   
        print(err_str)
        return None

    t1 = time.time()

    ###================world summaries================###

    world_df = logger.world_summary()

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

    world_df.to_csv(world_summary_path)
    daily_world_df.to_csv(daily_world_summary_path)
    
    print(f'save world summary {index}')    



    ###===============strat age summary==============###

    # some of the pandas commands could be suuuuuper dodgy

    strat_age_df = logger.age_summary(strat_age_bins)
    strat_age_df.index = pd.to_datetime(strat_age_df.index)  
    strat_age_df.set_index(['age_range',strat_age_df.index],inplace=True)

    total_pop_cols = ['infected', 'recovered', 'dead', 'susceptible']
    age_pops = strat_age_df.groupby('age_range')[total_pop_cols].nth(0).sum(axis=1)


    strat_age_df['new_hospital_dead'] = pd.Series(index=strat_age_df.index)
    strat_age_df['seroprevalence'] = pd.Series(index=strat_age_df.index)
    # set as empty series, with same index.

    for jj,(age_range,gdf) in enumerate(strat_age_df.groupby('age_range')):
        # gdf is a multi-index, with Index level0='0_5', level1=[list of timestamps]
        # can use subgroup index as "complete indexing" to set on larger df.

        sero = 100.*gdf['new_infections'].cumsum()/age_pops.iloc[jj]
        strat_age_df.loc[gdf.index,'seroprevalence'] = sero

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

        strat_age_df.loc[gdf.index,'new_hospital_dead'] = nhd.values.astype(int)
        # end of "for" loop - noted bc. I was confused by all the indention
        

    # undo the MultiIndex...
    strat_age_df.reset_index(level=0,inplace=True)

    # now do the resampled version!
    keep_cols = [
        'new_hospital_dead',
        'new_infections',
        'hospital_admissions',
        'intensive_care_admissions',
        'age_range'
    ]

    # can't just copy seroprevalence, as it's a cumulative.
    
    daily_strat_age_df = strat_age_df[keep_cols].groupby('age_range').resample('D').sum()
    daily_strat_age_df['seroprevalence'] = pd.Series(index=daily_strat_age_df.index)
    # set as empty series, with same index.

    # no need to recalc. age_pops

    for jj,(age_range,gdf) in enumerate(daily_strat_age_df.groupby('age_range')):
        # gdf is a multi-index, with Index level0='0_5', level1=[list of timestamps]
        # can use subgroup index as "complete indexing" to set on larger df.
        sero = 100.*gdf['new_infections'].cumsum()/age_pops.iloc[jj]
        daily_strat_age_df.loc[gdf.index,'seroprevalence'] = sero

    # undo the MultiIndex...
    daily_strat_age_df.reset_index(level=0,inplace=True)

    strat_age_df.to_csv(strat_age_summary_path)
    daily_strat_age_df.to_csv(daily_strat_age_summary_path)

    print(f'save strat age summary {index}')


    
    ###============save the hospital info===========###

    hospital_df = logger.load_hospital_capacity()
    hospital_df.to_csv(hospital_df_path)



    ###=========dump the parameters as json========###

    json_path = f'{summary_dir}/parameters_{index}.json'
    params.iloc[ii].to_json(json_path)

    t2 = time.time()#

    print(f'{index}\' s summary files saved in {(t2-t1)/60.} min')

    #print(f'{ii}/{250},iteration {iteration} done')

    return None


#do_extract(0)

if __name__ == "__main__":  
    
    parser = OptionParser()
    parser.add_option('--iteration',action='store',default=1)
    
    (options,args) = parser.parse_args()

    print('test "do_extract" on 0')
    """

    work_dir = f'/cosma5/data/durham/dc-sedg2/covid'

    do_extract(0,f'{work_dir}/june_results/r11_policytests/Run_1/policy_q14_lm8',
                summary_dir = f'{work_dir}/june_results/r11_policytests/summaries/Run_1/',
                iteration=1
    )
        
    do_extract(1,f'{work_dir}/june_results/r11_policytests/Run_1/policy_q14_lm2',
                summary_dir = f'{work_dir}/june_results/r11_policytests/summaries/Run_1/',
                iteration=1
    )
    

    do_extract(2,f'{work_dir}/june_results/r11_policytests/Run_1/policy_q0_lm2',
                summary_dir = f'{work_dir}/june_results/r11_policytests/summaries/Run_1/',
                iteration=1
    )
    
    do_extract(3,f'{work_dir}/june_results/r11_policytests/Run_1/policy_q7_lm4',
                summary_dir = f'{work_dir}/june_results/r11_policytests/summaries/Run_1/',
                iteration=1
    )
    

    """















