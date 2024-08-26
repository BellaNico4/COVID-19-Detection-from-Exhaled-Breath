import pandas as pd
from tqdm import tqdm
import sys
import os
import numpy as np
import re
from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.signal import detrend
from scipy.stats import zscore


def find_plateaus(F, min_length=3, max_length=7, tolerance=0.75, smoothing=1):
    '''
    Finds plateaus of signal using second derivative of F.

    Parameters
    ----------
    F : Signal.
    min_length: Minimum length of plateau.
    tolerance: Number between 0 and 1 indicating how tolerant
        the requirement of constant slope of the plateau is.
    smoothing: Size of uniform filter 1D applied to F and its derivatives.

    Returns
    -------
    plateaus: array of plateau left and right edges pairs
    dF: (smoothed) derivative of F
    d2F: (smoothed) Second Derivative of F
    '''
    # calculate smooth gradients
    #smoothF = uniform_filter1d(F, size=smoothing)
    #dF = uniform_filter1d(np.gradient(smoothF), size=smoothing)
    # plateau are zone in which first derivative of the function is almost flat

    #d2F = uniform_filter1d(np.gradient(dF),size = smoothing)
    dF = np.gradient(F)
    #d2F = np.gradient(dF)

    def zero_runs(x):
        '''
        Helper function for finding sequences of 0s in a signal
        https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
        '''
        # from bolean to int
        x = [0 if t == True else 1 for t in x]
        iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    # Find ranges where first derivative is almost zero

    # Values under eps are assumed to be zero.
    eps = np.quantile(abs(dF), tolerance)
    smalld2F = (abs(dF) <= eps)

    # Find repititions in the mask "smalld2F" (i.e. ranges where d2F is constantly zero)
    p = zero_runs(smalld2F)
    # np.diff(p) gives the length of each range found.
    # only accept plateaus of min_length
    plateaus = p[(np.diff(p) >= min_length).flatten()]
    #plateaus = plateaus[(np.diff(plateaus) <= max_length).flatten()]
    return (plateaus, dF, eps)


def get_stable_spectrum(df, features, window=3,tolerance=0.75,return_list = None):
    try:
        plateaus, dF, eps = find_plateaus(df[features].sum(
            axis=1,numeric_only=True), min_length=2, max_length=15, tolerance=tolerance, smoothing=1)
        if len(plateaus) == 0:
            # no plateau found...let's plot the mass spectra
            raise ValueError
        max_plat = plateaus[np.argmax(plateaus[:, 1] - plateaus[:, 0])]
        idx_min, idx_max = max_plat

        # if the plateau start at zero, it should be related to the acquisition with closed Valve
        if idx_min == 0:
            plateaus = plateaus[1:]
            if len(plateaus) == 0:
                # no plateau found...let's plot the mass spectra
                raise ValueError
            max_plat = plateaus[np.argmax(plateaus[:, 1] - plateaus[:, 0])]
            idx_min, idx_max = max_plat

        # find acquistion in the plateau that minimize tha std by mean of a rolling window
        std_min = df.iloc[idx_min:idx_max][features].sum(
            axis=1,numeric_only=True).rolling(window).std().idxmin()
        idx = np.where(df.index.values == std_min)[0][0]
        if return_list is not None:
            return_list.extend(df.iloc[idx-window:idx].index)
            #return_list.extend(df.iloc[idx-window+1:idx+1].index)
        return df.iloc[idx-window:idx][features].mean(numeric_only=True)
        #return df.iloc[idx-window+1:idx+1][features].mean(numeric_only=True)
    except:
        # plot the mass spectra to check why things went wrong
        display(df)
        fix, ax = plt.subplots(figsize=(20, 12))
        df.reset_index()[features].sum(axis=1).plot(ax=ax)
        plt.plot(dF)
        plt.hlines(eps, xmin=0, xmax=len(dF))
        plt.hlines(-eps, xmin=0, xmax=len(dF))
        plt.show()
        print(plateaus)
        #print(max_plat)
        #print(idx_min, idx_max)
        #print(std_min)
        #print(df.iloc[idx_min:idx_max][features].sum(
            #axis=1).rolling(window).std())


def get_files_by_ext(path: str, extension: str, rename: bool = False, prefix: str = "pazienti") -> list:
    ext_paths = []
    directories = 0
    print(
        f"Getting {extension} files files starting from {path}. (Rename={rename}) ...")
    for x in tqdm(os.walk(path)):
        # get all the direcories in the path folder
        dir_path = x[0]
        directories += 1

        for _file in os.listdir(dir_path):
            # obtain the full path of the file
            file_path = f"{dir_path}/{_file}"
            if os.path.isfile(file_path):
                # date = directory name
                date = dir_path.split("/")[-1]
                ext = _file.split(".")[-1]
                if ext == extension:
                    # check file format name and rename it if the parameter is set to true
                    if _file != f"{prefix}-{date}.{ext}" and rename:
                        new_file_path = f"{dir_path}/{prefix}-{date}.{ext}"
                        os.rename(file_path, new_file_path)
                        file_path = new_file_path
                    ext_paths.append(file_path)
    print(
        f"\t\t--- Found [{len(ext_paths)}] {extension} files. Visited {directories-1} directories.")
    return ext_paths


def df_from_ASC_file(filepath: str, verbose: bool = False) -> tuple[list, list]:
    """from an ASC file return the amus list and a list of measures.
    Args:
        filepath (str): path of the ASC file
        verbose (bool, optional): Print the message during the file reading. Defaults to False.

    Returns:
        tuple[list, list]: a tuple with a list of list of measures (each list is relative to a time of acquisition) and 
        a list of AMUs
    """
    if not os.path.isfile(filepath):
        print(f"{filepath} is not a valid file.")
        exit(-1)
    # line that delimits the measures
    SPLIT_LINE = "IntervalOfScan="
    # return values 
    all_measures = []
    amus = []

    if verbose:
        print(f"FROM [df_from_ASC_file(...)] building a df : path={filepath}")

    with open(filepath, "r") as asc_file:
        content = asc_file.read()
        for fold in content.split(SPLIT_LINE):
            # each fold contains all the measures in the format AMU     intensity\n
            if fold.startswith("##"):
                continue
            # see previous comment
            measures = fold.split('\n')
            # first value after interval of scan is the acquisition time
            toa = float(measures[0])
            # remove the toa
            del measures[0]

            # filter "" values
            measures = list(filter(lambda elm: elm, measures))
            # split amu and measure and get the 2 lists
            amus, measures = zip(
                *[re.split("\s+", measure.strip(' ')) for measure in measures])
            # convert both list to a proper format
            amus = [f'{float(amu):.2f}' for amu in amus]
            measures = [float(m) for m in measures]
            
            all_measures.append(measures)

    return all_measures, amus


def build_raw_dataframe(base_path: str, range_num: int, verbose: bool = False) -> pd.DataFrame:
    """Given a base path in which are contained all the folders that contains the ASC files return a dataframe of the specified
    range with the following structure: 'key', 'index', AMU1, ..., AMUn. The key is composed by the date of the measure, the patient
    number and the acquisition relative to the ASC file.  

    Args:
        base_path (str): path of the parent directory of the acquisitions folders
        range_num (int): range to consider i.e 'date patient_range.ASC'
        verbose (bool, optional): if set print the step by step messages of the script. Defaults to False.

    Returns:
        pd.DataFrame: the dataframe with all the raw measures taken.
    """
    asc_paths = get_files_by_ext(base_path, 'ASC')
    df_rows = []
    keys = []
    amus = []
    for asc_filepath in asc_paths:
        date = asc_filepath.split("/")[-2]
        # take the measure information from the ASC filename: i.e "jul 5 2021 1_0.ASC" it splits over the space to get the measure
        # and the extension and with another split on the "." it get the 1_0 (patient 1, measure 0)
        asc_filename = asc_filepath.split(".")[0].strip(' ')
        
        measure_id = asc_filename.split(
            " ")[-1] if "bis" not in asc_filepath else asc_filename.split(" ")[-2].strip(' ')

        measure_info = measure_id.split("_")
        try:
            if measure_info[-2] == "0":
                if verbose:
                    print(f'Air Sample in file: {asc_filepath}')
                continue
        except IndexError:
            print(measure_info, asc_filename)
        # from now on each file should have the measure id as patient_range since we have filtered out the air measures
        try:
            patient_number, range_ = measure_info
        except ValueError:
            print('Error in splitting patient number and range, skipping...', measure_info, asc_filepath)
            continue
        if int(range_) != range_num:
            continue

        # a usefull column to group by date and patient
        patient_id = f"{date}_{patient_number}"
        # get a list of measure contained in the asc file, each time of acquisition(toa) has a list with all the relative measures
        # toas is a parallel vector of file_measures i.e measures = [[100, 21, 123, 21 ...], [123, 121, 4, 54, ...]] toas =[5, 10]
        # the former list is the measure relative to the 5s time acquisition
        file_measures, amus = df_from_ASC_file(asc_filepath)

        # resulting dataframe structure: date_patient_range_Nacquisition, AMUs, patient_id (external key of the patient dataframe with
        # patient information, in practice is the the date concatenate with the measure)
        for n_acq, measures in enumerate(file_measures):
            # primary key column
            key = f"{date}_{measure_id}_acq{n_acq}"
            keys.append(key)
            # build the row of the dataframe, they are made of the patient id (date + id) and the amus intensities
            row = [patient_id] + measures 
            df_rows.append(row)
    columns = ["index"] + amus
    df = pd.DataFrame(np.vstack(df_rows), columns=columns, index=keys)
    df[amus] = df[amus].values.astype('I')
    
    df[amus] = df[amus].apply(np.floor)
    df[amus] = df[amus].values.astype('I')
    return df


def create_csv(base_path: str, range_: int, save_path:str):
    df = build_raw_dataframe(base_path, range_)
    df.to_csv(save_path)

def get_stable_spectra(save_path:str, spectra_to_keep:int=4) -> pd.DataFrame:
    """Given a base path in which are contained all the folders that contains the ASC files return a dataframe of the specified
    range with the following structure: 'index', AMU1, ..., AMUn. 
    The index is the primary key of the dataframe and is composed of the date of the measure and the patient identifier. The returned
    dataframe is the result of the picking of spectra_to_peak spectra, this paramater specifies the number of spectra to take from
    the plateau of the TIC plot of the relative measure (Default = 4). 
    The resulting dataframe will be save in csv format in the save_path file.

    Args:
        range_ (int): range to consider i.e 'date patient_range.ASC'
        save_path (str): where to save the obtained dataframe
        spectra_to_keep (int): number of acquisition on the flat-zone to average
    Returns:
        pd.DataFrame: the resulting dataframe after the application of the stable spectra picking algorithm.
    """
    df = pd.read_csv(save_path)
    df.drop(columns=['date','id','index_','full name','covid','healed'],inplace=True,errors='ignore')
    amus = list(df.columns.drop(['index','acq'],errors='ignore'))
    df = df.groupby('index').apply(get_stable_spectrum, amus, spectra_to_keep).reset_index()
    
    return df

def normalize_spectrum_peak(row):
    max_intensity = np.max(row)
    return row.div(max_intensity)
def normalize_spectrum_sum(row):
    tic = np.sum(row)
    return row.div(tic)


def get_labels(base_path: str, range_: int) -> pd.DataFrame:
    df_labels = pd.read_csv(base_path)
    df_labels = df_labels.drop(columns=['Mass Range','Unnamed: 0','Patient'])
    
    def create_patients_id(row):
        row['index'] = ''.join(row['Date'].split('-')) + '_' + row['File'].split('_')[0]
        return row

    def create_range_columns(row):
        row['Range'] = row['File'].split('_')[1]
        return row

    df_labels = df_labels.apply(create_patients_id,axis=1)
    df_labels = df_labels.apply(create_range_columns,axis=1)
    df_labels['Range'] = pd.to_numeric(df_labels['Range'])
    df_labels = df_labels.dropna(subset=['Covid'])
    df_labels['Healed'] = df_labels['Healed'].fillna(0)
    df_labels['CovidDate'] = df_labels['CovidDate'].fillna('')
    df_labels['Comment'] = df_labels['Comment'].fillna('')


    df_labels = df_labels[df_labels['Range'] == range_]
    #Validity Check#
    df_labels = df_labels[df_labels['Validity(1=Valid;0=NonValid)'] == 1]
    
    
    df_labels = df_labels[df_labels['Comment'].str.contains("TIC|DISCONESSO")==False]
    df_labels['Date'] = pd.to_datetime(df_labels['Date'])
    df_labels['CovidDate'] = pd.to_datetime(df_labels['CovidDate'],errors='coerce')
    df_labels['Covid Days'] = df_labels['Date'] - df_labels['CovidDate']
    
    df_labels['Covid'] = df_labels['Covid'].astype(int)
    df_labels['Healed'] = df_labels['Healed'].astype(int)
    df_labels['Covid_Healed'] = df_labels['Covid'] | df_labels['Healed']
    return df_labels
    
def get_acquisitions_in_plateau(spectra_path: str,range_idx,spectra_to_keep:int=4,tolerance=0.75) -> pd.DataFrame:
    """        
    return_list (list): list in which eventually save the integer index to retrieve the acquisition in the plateau zone, for each patient

    """
    df = pd.read_csv(spectra_path, index_col=0).reset_index(names='acq')
    df.drop(columns=['date','id','index_','full name','covid','healed'],inplace=True,errors='ignore')
    
    amu_ranges = {
    0: ('10.0', '51.0'),
    1: ('10.0', '51.0'),
    2: ('49.0', '151.0'),
    3: ('149.0', '251.0'),
    4: ('249.0', '351.0')
    }
    try:
        idx_min = df.columns.get_loc(amu_ranges[range_idx][0])
        idx_max = df.columns.get_loc(amu_ranges[range_idx][1])
    except KeyError:
        amu_ranges = {
        0: (10.0, 51.0),
        1: (10.0, 51.0),
        2: (49.0, 151.0),
        3: (149.0, 251.0),
        4: (249.0, 351.0)
        }
        idx_min = df.columns.get_loc(amu_ranges[range_idx][0])
        idx_max = df.columns.get_loc(amu_ranges[range_idx][1])
        
    features = list(df.columns[idx_min:idx_max+1])
    index_list=[]
    
    df.groupby('index').apply(get_stable_spectrum,features,window=spectra_to_keep,return_list=index_list,tolerance=tolerance)
    
    df_multiple_spectra = df.iloc[index_list]
    return df_multiple_spectra


def apply_filters(df,features): 
    filtered_df =  df[(np.abs(zscore(df[features])) < 8).all(axis=1)]
    filtered_df.loc[:,features] = filtered_df[features].where(filtered_df[features] > 0.0001, 0).values
    #filtered_df[features] = filtered_df[features].apply(detrend,axis=1,raw=True,type='constant')
    filtered_df.loc[:,features] = filtered_df[features].apply(savgol_filter,axis=1,raw=True,window_length=5, polyorder=2, deriv=0).values
    filtered_df.loc[:,features] = filtered_df[features].where(filtered_df[features] > 0.001, 0).values
    return filtered_df