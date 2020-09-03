import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.backends.backend_pdf

import time
 
plt.style.use(['science', 'notebook'])
plt.style.reload_library()

from june.logger.read_logger import ReadLogger
from june.policy import Policy, Policies, SocialDistancing

from .extract_data_new import *

class Plotter():

    def __init__(self, logger, real_data_path, summary_dir, parameter_index):
        self.logger = logger
        self.summary_dir = summary_dir
        self.plot_save_dir = summary_dir / 'plots'
        if os.path.exists(self.plot_save_dir) is False:
            os.mkdir(self.plot_save_dir)
        self.parameter_index = parameter_index
        self.policies = Policies.from_file()
        self.total_pop_cols = ['current_infected',
                               'current_recovered',
                               'current_dead',
                               'current_susceptible']
        self.real_data_path = real_data_path
        self.real_data = self.check_real_data()
        if self.real_data:
            self.admissions_df, self.deaths_df = self.load_real_data()
        self.regional_run_summary = self.create_regional_summaries()
        self.daily_age_summary, self.age_labels = self.create_age_summaries()

    def check_real_data(self):
        self.admissions_data_path = self.real_data_path / 'new-sitrep-admissions-v2-mean.csv'
        self.deaths_data_path = self.real_data_path / 'regional_deaths_5year_strat.csv'
        if os.path.exists(self.admissions_data_path) and os.path.exists(self.deaths_data_path):
            return True
        else:
            return False

    def load_real_data(self):
        # Tristan's bootstrapped data
        admissions_data = pd.read_csv(self.admissions_data_path, index_col='Unnamed: 0')
        admissions_regions = admissions_data['Region'].unique()
        admissions_bands = admissions_data['Band'].unique()

        deaths_data = pd.read_csv(self.deaths_data_path)
        deaths_data = deaths_data.melt(id_vars=['region_name', 'date_of_death'], var_name='Band', value_name='Deaths')
        deaths_regions = deaths_data['region_name'].unique()
        deaths_bands = deaths_data['Band'].unique()

        # make data more easily plottable
        admissions_df = {}
        for band in admissions_bands:
            dfs = {}
            for region in admissions_regions:
                tmp_df = admissions_data[(admissions_data['Band']==band) & (admissions_data['Region']==region)].filter(items=['Date', 'Admissions'])
                tmp_df.index = pd.to_datetime(tmp_df['Date'])
                tmp_df.drop(columns=['Date'], inplace=True)
                if region == 'East Of England':
                    region = region.replace('Of', 'of')
                elif region == 'North East And Yorkshire':
                    region = region.replace('And', 'and')
                dfs[region] = tmp_df
            admissions_df[band] = pd.concat({key: val for key, val in dfs.items()})
        admissions_df = pd.concat({key: val for key, val in admissions_df.items()})

        deaths_df = {}
        for band in deaths_bands:
            dfs = {}
            for region in deaths_regions:
                tmp_df = deaths_data[(deaths_data['Band']==band) & (deaths_data['region_name']==region)].filter(items=('date_of_death', 'Deaths'))
                tmp_df.index = pd.to_datetime(tmp_df['date_of_death'])
                tmp_df.drop(columns=['date_of_death'], inplace=True)
                if region == 'East Of England':
                    region = region.replace('Of', 'of')
                dfs[region] = tmp_df
            dfs['All regions'] = pd.concat({key: val for key, val in dfs.items()}).sum(level=1).sort_index()
            deaths_df[band] = pd.concat({key: val for key, val in sorted(dfs.items())})
        deaths_df['All ages'] = pd.concat({key: val for key, val in deaths_df.items()}).sum(level=[1,2]).sort_index()
        deaths_df = pd.concat({key: val for key, val in deaths_df.items()})

        return admissions_df, deaths_df
    
    def create_regional_summaries(self):
        # TODO link up with csvs
        run_summary = self.logger.run_summary()
        regional_run_summary = run_summary.groupby(['region', run_summary.index]).sum()
        all_regions = regional_run_summary.index.get_level_values(0).unique()

        # group Midlands together and North East and Yorkshire
        regional_run_summary = group_all_region_df(regional_run_summary, all_regions)

        # get region names and time stamps for plotting
        self.plotting_regions = regional_run_summary.index.get_level_values(0).unique()
        self.time_stamps = regional_run_summary.index.get_level_values(1).unique()

        # calculate region populations
        region_populations = {region: regional_run_summary[self.total_pop_cols].loc[region].iloc[0].sum() for region in self.plotting_regions}

        for region in self.plotting_regions:
            seroprev_df = 100. * regional_run_summary.loc[region, 'daily_infections'].cumsum() / region_populations[region]
            seroprev_df.index = pd.MultiIndex.from_product([[region], regional_run_summary.loc[region].index.get_level_values(0)])
            regional_run_summary.loc[region, 'seroprevalence'] = seroprev_df

        return regional_run_summary

    def create_age_summaries(self):
        # TODO link up with csvs
        ages_df = self.logger.age_summary(np.arange(0, 101))

        # create daily summaries of daily columns
        dfs = []
        for age, group in ages_df.groupby('age_range'):
            tmp_df = group.filter(regex="daily_*").resample('D').sum()
            tmp_df.insert(0, 'age', int(age.split('_')[0]))
            dfs.append(tmp_df)
        daily_age_df = pd.concat(dfs)

        # currently plot in 5 year age bins
        age_range = np.arange(0, 100+1, 5)
        labels = ['_'.join((str(age_range[i]+1), str(age_range[i+1]))) for i in range(len(age_range)-1)]
        labels[0] = '0_5'
        # assign age ranges
        daily_age_df['age_range'] = pd.cut(daily_age_df.age, bins=age_range, labels=labels, include_lowest=True)
        world_ages = self.logger.ages
        # population of each age range
        age_populations = {labels[i]: len(world_ages[(world_ages > age_range[i]) & (world_ages <= age_range[i+1])]) for i in range(len(age_range)-1)}

        # group by age ranges
        dfs = {}
        for age_range, group in daily_age_df.groupby('age_range'):
            tmp_df = group.drop(columns=['age', 'age_range'])
            tmp_df = tmp_df.groupby('time_stamp').sum()
            tmp_df['seroprevalence'] = 100. * tmp_df['daily_infections'].cumsum() / age_populations[age_range]
            dfs[age_range] = tmp_df
        daily_age_summary = pd.concat({key: val for key, val in dfs.items()})

        return daily_age_summary, labels

    def plot_region_data(self):
        fig, ax = plt.subplots(4, len(self.plotting_regions), figsize=(6*len(self.plotting_regions), 15), sharex=True)
        plt.suptitle('Run summaries for regions of England', fontsize=24)
        for i, region in enumerate(sorted(self.plotting_regions)):
            ax[0, i].set_title(region)
            ax[0, i].plot(self.regional_run_summary.loc[region]['daily_infections'])
            ax[1, i].plot(self.regional_run_summary.loc[region]['daily_hospital_admissions'], label='june')
            ax[2, i].plot(self.regional_run_summary.loc[region]['daily_deaths_hospital'] \
                        + self.regional_run_summary.loc[region]['daily_deaths_icu'], label='june')
            ax[3, i].plot(self.regional_run_summary.loc[region]['seroprevalence'])
            
            if self.real_data:
                # doesn't always work depending on regions available in data...
                try:
                    ax[1, i].plot(self.admissions_df.loc['Admissions_Total', region], label='data', color='k')
                except:
                    continue
                try:
                    ax[2, i].plot(self.deaths_df.loc['All ages', region], label='data', color='k')
                except:
                    continue
            
        self.format_axes(ax)
        ax[0, 0].set_ylabel('Daily infections')
        ax[1, 0].set_ylabel('Daily \n hospital admissions')
        ax[2, 0].set_ylabel('Daily \n hospital deaths')
        ax[3, 0].set_ylabel('Percentage \n seroprevalence')
        for axes in ax[1:3].ravel():
            axes.legend()
        for axes in ax[3]:
            axes.yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.subplots_adjust(top=0.90)
        plt.savefig(self.plot_save_dir / f'region_plots_{self.parameter_index:03}.pdf', dpi=300, bbox_inches='tight')
        return None

    def plot_age_stratified(self):
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.plot_save_dir / f'age_plots_{self.parameter_index:03}.pdf')

        # hospital admissions plot
        fig, ax = plt.subplots(4, 5, figsize=(28, 16), sharex=True)
        plt.suptitle('Age stratified daily hospital admissions for England', fontsize=24)
        for i, axes in enumerate(ax.ravel()):
            axes.set_title(self.age_labels[i])
            axes.plot(self.daily_age_summary.loc[self.age_labels[i]]['daily_hospital_admissions'], label='june')
            axes.legend()

        self.format_axes(ax)
        for axes in ax.T[0]:
            axes.set_ylabel('Daily \n hospital admissions')
        plt.subplots_adjust(top=0.90)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')

        # daily deaths plot
        fig, ax = plt.subplots(4, 5, figsize=(28, 16), sharex=True)
        plt.suptitle('Age stratified daily hospital deaths for England', fontsize=24)
        for i, axes in enumerate(ax.ravel()):
            axes.set_title(self.age_labels[i])
            axes.plot(self.daily_age_summary.loc[self.age_labels[i]]['daily_deaths_hospital'] + \
                      self.daily_age_summary.loc[self.age_labels[i]]['daily_deaths_icu'], label='june')
            if self.real_data:
                # doesn't always work depending on regions available in data...
                try:
                    axes.plot(self.deaths_df.loc[self.age_labels[i].replace('_', '-')].sum(level=1).sort_index(),
                    label='data', color='k')
                except:
                    continue
            axes.legend()
        
        self.format_axes(ax)
        for axes in ax.T[0]:
            axes.set_ylabel('Daily \n hospital deaths')
        plt.subplots_adjust(top=0.90)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')

        # seroprevalence plot
        fig, ax = plt.subplots(4, 5, figsize=(28, 16), sharex=True)
        plt.suptitle('Age stratified seroprevalence for England', fontsize=24)
        for i, axes in enumerate(ax.ravel()):
            axes.set_title(self.age_labels[i])
            axes.plot(self.daily_age_summary.loc[self.age_labels[i]]['seroprevalence'])
        
        self.format_axes(ax)
        for axes in ax.T[0]:
            axes.set_ylabel('Percentage \n seroprevalence')
        plt.subplots_adjust(top=0.90)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        
        pdf.close()
        return None

    def format_axes(self, ax):
        for axes in ax.ravel():
            axes.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.MO))
            for policy in self.policies.policies:
                axes.axvspan(min(max(policy.start_time, self.time_stamps.min()), self.time_stamps.max()),
                            min(policy.end_time, self.time_stamps.max()),
                            alpha=0.01)
            axes.axvline(datetime(2020, 3, 23, 0, 0),
                        linestyle='dashed',
                        color='indianred')
            axes.set_xlim([self.time_stamps.min(), self.time_stamps.max()])
            for tick in axes.get_xticklabels():
                tick.set_rotation(45)
        return None

if __name__ == "__main__":
    t0 = time.time()
    logger_path = Path('/home/htruong/Documents/JUNE/Notebooks/results')
    real_data_path = Path('/home/htruong/Documents/JUNE/Notebooks/')
    save_path = Path('./')
    logger = ReadLogger(logger_path)
    plotter = Plotter(logger, real_data_path)
    plotter.plot_region_data(save_path)
    plotter.plot_age_stratified(save_path)
    print("Total time = {:.1f}s".format(time.time() - t0))
