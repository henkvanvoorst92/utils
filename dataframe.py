
import pandas as pd
import numpy as np
import os
from datetime import datetime

def stata_reader(path, conv_cat=False):
	df = pd.read_stata(path, convert_categoricals = conv_cat)
	print('Number of variables (columns):', len(df.columns))
	print('Number of rows:', len(df))
	return df

def store_df(input, sav_loc, tpe, cols=None, separate=None):
    if isinstance(input,list):
        df_out = pd.DataFrame(input).reset_index(drop=True)
    elif isinstance(input,pd.DataFrame):
        df_out = input
    else:
        print('input type not compatible',type(input))
        raise AssertionError('wrong data type')
    

    if isinstance(cols,list):
        df_out.columns = cols
    if separate is not None:
        sav_loc+='_'+str(separate)
        
    sav_loc += tpe
    if os.path.exists(sav_loc):
        if tpe=='.xlsx':
            tmp = pd.read_excel(sav_loc)
        elif tpe=='.pic':
            tmp = pd.read_pickle(sav_loc)
        elif tpe=='.ftr':
            tmp = pd.read_feather(sav_loc)
        else:
            print('Error wrong type of extension:', tpe)
        if tmp.shape != (0,0): #sometimes written tmp is empty
            if 'Unnamed: 0' in tmp.columns:
                tmp = tmp.drop(columns=['Unnamed: 0'])
            if isinstance(cols,list):
                tmp.columns = cols
            df_out = pd.concat([df_out,tmp], axis=0).reset_index(drop=True)

    if tpe=='.xlsx':
        df_out.to_excel(sav_loc)
    elif tpe=='.pic':
        df_out.to_pickle(sav_loc)
    elif tpe=='.ftr':
        df_out.to_feather(sav_loc)

def wide2long(df_wide, colID, rater_cols, 
              rater_colname, other_cols=None):
    #converts wide dataset to long format
    out = []
    for ix,rc in enumerate(rater_cols):
        tmp = pd.DataFrame(df_wide[rc].values, 
                           columns=[rater_colname], 
                           index= df_wide[colID])
        tmp['rater'] = rc
        if other_cols is not None:
            tmp[other_cols] = df_wide[other_cols]
        tmp = tmp.reset_index()
        out.append(tmp)
    
    return pd.concat(out)

def excel_multtabs(dfs, tabnames, fname, loc):
    with pd.ExcelWriter(os.path.join(loc,fname+'.xlsx')) as writer:
        
        for c,file in enumerate(dfs):
            file.to_excel(writer, sheet_name=tabnames[c])
    return

def percentile_vars(df,varname, top_bottom_p=10):
    var = df[varname]
    # compute bottom and top percentile values
    p_bottom = np.percentile(var.dropna(), top_bottom_p)
    p_top = np.percentile(var.dropna(), 100-top_bottom_p)
    print('Lower-upper percentile:',int(p_bottom), int(p_top))
    # create new variables that split at percentiles
    df[varname+'_'+str(int(p_bottom))] = (var<p_bottom)*1
    df[varname+'_'+str(int(p_bottom))+'_'+str(int(p_top))] = \
    ((var>=p_bottom)&(var<=p_top))*1
    df[varname+'_'+str(int(p_top))] = (var>p_top)*1
    
    # put new variables in a list
    new_vars = [varname+'_'+str(int(p_bottom)), 
                  varname+'_'+str(int(p_bottom))+'_'+str(int(p_top)),
                  varname+'_'+str(int(p_top))]
    # make a categorical variable of the grouped variables
    df[varname +'_cat'] = np.repeat(np.NaN,df.shape[0])
    for c in new_vars:
        df[varname+'_cat'][df[c]==1]=c
        df[c][var.isna()] = np.NaN
    return df

# extension of percentile_vars but now for all bins and not only lower-upper
def percentile_dummies(df, varname, steps=10, inbetween=True):
    var = df[varname]
    # iterate over the values of trombo counts
    pctile_dct = {}
    newvars = [varname]
    for p in range(steps,100, steps):
        # get the value of the corresponding percentile
        pctile = np.percentile(var.dropna(), p)
        pctile_dct[p] = pctile
        
        # operation for lowest percentile
        if p==steps:
            dum = (var<=pctile)*1
            dumname = varname+'_p'+str(p)
            df[dumname] = dum
            newvars.append(dumname)
        # operation for all other percentiles
        else:
            # operation for all non maxima percentiles
            if inbetween==True:
                dum = ((var>pctile_past)&(var<=pctile))*1
                dumname = varname+'_p'+str(p_past)+'-'+str(p)
                df[dumname] = dum
                newvars.append(dumname)
            # operations for the upper percentile
            if p==(100-steps):
                dum = (var>pctile)*1
                dumname = varname+'_p'+str(p)
                df[dumname] = dum
                newvars.append(dumname)

        pctile_past = pctile.copy()
        p_past = p
    df[varname +'_cat'] = np.repeat(np.NaN,df.shape[0])
    for c in newvars[1:]:
        df[varname+'_cat'][df[c]==1]=c
        df[c][var.isna()] = np.NaN

    newvars.append(varname +'_cat')

    return df, newvars, pctile_dct

def dt2float(arr):
    out =[]
    for i in arr:
        if isinstance(i, (datetime, pd._libs.tslibs.timedeltas.Timedelta, 
            pd._libs.tslibs.timedeltas.Timedelta,np.datetime64)):
            out.append((i.seconds+i.microseconds/1e6)*1e5)
        else:
            out.append(i)          
    return out

def multi_stats(data, stat_list=None):
    # stat_list is a list of np operations (example: np.mean --> 'mean')
    # applied to an array
    if stat_list is None:
        stat_list = ['mean', 'median', 'std', 'percentile', 'min', 'max']
    out = []
    for st in stat_list:
        if st=='percentile':
            for p in [5,10,25,75,90,95]:
                pctile = np.percentile(data,p)
                out.append(['p'+str(p),pctile])
        else:
            oper = getattr(np, st)
            out.append([st,oper(data)])
    return pd.DataFrame(out)


