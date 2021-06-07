import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy

from sklearn.metrics import mean_squared_error, r2_score
import scipy
from scipy import stats


def hellojuju():
    print('Hello juju')

def dichotomize_dataset(dataset, columnsToNotDicho):
    dichotomizeDF = pd.DataFrame()
    for column in dataset :
        if(column not in columnsToNotDicho):
            dummies = pd.get_dummies(dataset[column], prefix=column)
            dummies.reset_index(drop=True, inplace=True)
            dichotomizeDF.reset_index(drop=True, inplace=True)
            dichotomizeDF = pd.concat([dichotomizeDF, dummies], axis=1, sort=True)
        else:
            dichotomizeDF[column] = dataset[column]
    return dichotomizeDF


def discretise_1col_quali(col, colname, regle):
    newCol = []
    newColPandas = pd.DataFrame()
    ligne = 0 
    errors = 0
    for value in col:
        found = 0
        for key in regle:
            if(found ==0 ):
                for remplacement in regle[key]:
                    if (str(value) == remplacement):
                        newCol.append(key)
                        found=1
        if(found==0):
            print( "Variable : " + colname + " ligne : " + str(ligne) + " le programme ne trouve pas : '" + str(value) + "' dans les règles")
            errors +=1
        ligne +=1
    newColPandas =  newCol           
    if( len(col) != len(newColPandas)):
        print('erreur avec la variable : ' + colname + " veuillez vérifier la colone. (" + str(len(col)) + " != " + str(len(newColPandas)) + ')')    
        errors +=1
    if(errors != 0):
        return col
    else:
        return newColPandas

def discretise_1col_quanti(col, colname, regle):
    #Left cap excluded, right cap included
    newCol = []
    newColPandas = pd.DataFrame()
    ligne = 0 
    errors = 0
    for value in col:
        found = 0
        for key in regle:
            if(found ==0 ):
                borneBasse = regle[key][0]
                borneHaute = regle[key][1]
                try:
                    value = int(value)
                except Exception:
                    pass
                if(isinstance(value, int)):
                    if((value >= borneBasse) and (value <= borneHaute)):
                        newCol.append(key)  
                        found=1

        if(found==0):
            print( "Variable : " + colname + " ligne : " + str(ligne) + " le programme ne trouve pas : '" + str(value) + "' dans les règles")
            errors +=1
        ligne +=1
    newColPandas =  newCol           
    if( len(col) != len(newColPandas)):
        print('erreur avec la variable : ' + colname + " veuillez vérifier la colone. (" + str(len(col)) + " != " + str(len(newColPandas)) + ')')
        errors +=1
    if(errors != 0):
        return col
    else:
        return newColPandas



def runModels1DS(dataset, name, X, y, models):
    
    #This function take a dataset in entry
    #a name, like "train" or "test" 
    #a list of independent features X
    #a target feature y
    #An object containing models to use

    preds = pd.DataFrame()
    preds = copy.deepcopy(dataset)
    preds = preds.loc[:, ['Id', 'SalePrice', y]]
    #preds = preds.drop(preds.columns.difference(['Id',y]), 1, inplace=True)
    for model in models :
        
        pred = models[model]['function'].predict(dataset[X])
        preds['temp_' + name + model] = np.exp(pred.astype(float))
        preds['pred_' + name + '_' + model] = preds['temp_' + name + model].clip(lower=0)
        preds.drop(['temp_' + name + model], axis=1, inplace=True)
        
    return preds



def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2




def afficheResults(train_preds, test_preds, y, models):
    '''
    AfficheResults prend en entré deux dataframe (prévu pour un train et un test) et un nom de variable cible
    Chacun de ces dataframe contient des colonnes de prédiction (une par model) et la colonne des valeurs réel
    la fonction ne retourn rien mais affiche les coeff de corrélation / détermination et autres R² pour les deux dataframes
    '''
    for model in models:
        print('------------- ' + models[model]['label'] + ' -------------')
        print()

        print('Train : ', end="")

        print("R² = "+"{:.1%} ".format(r2_score(train_preds[y], train_preds['pred_train_' + model])), end="")
        #print(", rsquared = "+"{:.1%} ".format(rsquared(train_preds[y], train_preds['pred_train_' + model])), end="")
        #print(", corr = "+"{:.1%} ".format(train_preds[y].corr(train_preds['pred_train_' + model])), end="")
        print(', MSE = %.2f'
          % mean_squared_error(train_preds[y], train_preds['pred_train_' + model]))

        print('Test : ', end="")
        print("R² = "+"{:.1%} ".format(r2_score(test_preds[y], test_preds['pred_test_' + model])), end="")
        #print(", rsqaured = "+"{:.1%} ".format(rsquared(test_preds[y], test_preds['pred_test_' + model])), end="")
        #print(", corr = "+"{:.1%} ".format(test_preds[y].corr(test_preds['pred_test_' + model])), end="")
        print(', MSE = %.2f'
              % mean_squared_error(test_preds[y],  test_preds['pred_test_' + model]))
        print()

        '''
        plt.scatter(test_rwrk[y], test_rwrk['pred_test_' + modelList[item]])
        plt.plot([0, 600000], [0, 600000], color = 'red', linestyle = 'solid')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")

        print(rlm.intercept_)
        '''


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    # Source : https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn/24447#24447

    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            #best_feature = new_pval.argmin()
            best_feature = new_pval.keys()[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            #worst_feature = pvalues.argmax()
            worst_feature = pvalues.keys()[pvalues.argmax()]
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def corr_quali_quanti(x,y):
    #Attention eta_squared(x,y est réservé aux corrélations quali/quanti
    # x = qualitative
    # y = quantitative

    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT

def corr_quanti_quanti(x,y):
    #Cette fonction s'applique aux corrélations  Dicho/Quanti ou Quanti/Quanti
    return np.corrcoef(x, y)[0,1]



def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    #Cette fonction produit un graphique de distributions avec deux courbes, une pour le train et l'autre pour le test
    #RedFunction et BlueFunction sont deux arrays contenant des valeurs prédites et des valeurs constaté.
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Variable étudié')
    plt.ylabel('Nombre d\'individus concernés')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):

    #Cette fonction permet de supperposer deux nuages de points, le test et le train
    #Ces nuages de points sont constitué chacun d'une variable explicative et d'une expliqué
    #exemple ==> xtrain la taille d'une maison et y_train son prix. 
    #Une regression linéaire polynomial est alors utilisé et la courbe est inscrite dans le graphique. 
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object ex : lr = LinearRegression()
    #poly_transform:  polynomial transformation object ex : poly_transform = PolynomialFeatures(degree=2)
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


def f(order, test_data):
    #cette fonction permet d'avoir un graphique interactif pour tester différents polynomes et taille d'échantillon test
    # pour l'utilise avec cette commande : interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)




#---------------------------------- DISCRETISATION SUPERVISE -----------------------------
def discretisationSupervise(X_train, X_test, y_train, y_test, getPlot, getClasses, min_samples_split_percent):
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt

    X_train = X_train.copy(deep=True)
    X_test = X_test.copy(deep=True)
    dtrRsquared = pd.DataFrame(columns=['DTR' + str(min_samples_split_percent) + '_R²'])
    for var in X_train :            
        X = X_train[var].values
        rwrk_Xtrain = X.reshape(-1, 1)
        y = y_train.values
        rwrk_ytrain = y.reshape(-1, 1)

        Xtest = X_test[var].values
        rwrk_Xtest = Xtest.reshape(-1, 1)
        ytest = y_test.values
        rwrk_ytest = ytest.reshape(-1, 1)
        
        #Trouver max valid
        
        #res = list()
        res_valid = list()
        maxValid = [0,0]
        
        #Pourcentage d'occurence minimum dans une classe
        occurenceMin = round(len(X_train)*min_samples_split_percent/100)
        if (occurenceMin==0):
            occurenceMin=2
        for i in [2]:
            dtr = DecisionTreeRegressor(max_depth = i, min_samples_split = occurenceMin)
            dtr.fit(rwrk_Xtrain,rwrk_ytrain)
            pred_train = dtr.predict(rwrk_Xtrain)
            pred_test = dtr.predict(rwrk_Xtest)

            X_train[var +'_pred_train'] = pred_train.astype(int)
            X_test[var +'_pred_test'] = pred_test.astype(int)

            X_train[var + '_pred_train'] = X_train[var + '_pred_train'].clip(lower=0)
            X_test[var + '_pred_test'] = X_test[var + '_pred_test'].clip(lower=0)

            #res.append(rsquared(y_train, X_train[var + '_pred_train']))
            r2valid = rsquared(y_test, X_test[var + '_pred_test'])
            res_valid.append(r2valid)
            
            if (r2valid > maxValid[1]):
                maxValid[0] = i
                maxValid[1] = r2valid

        dtrRsquared.loc[var] = round(maxValid[1]*100,2)

        #Class selon DTR 
        tree_model = DecisionTreeRegressor(max_depth =maxValid[0],  min_samples_leaf = occurenceMin)
        tree_model.fit(rwrk_Xtrain,rwrk_ytrain)
        X_train[var + '_SPT']=tree_model.predict(rwrk_Xtrain)

        if(getPlot == True):
            #print(pd.concat( [X_train.groupby([var + '_SPT'])[var].min(), X_train.groupby([var + '_SPT'])[var].max()], axis=1))
            print(var + ' tree depth = ' + str(maxValid[0]) + ', R² = ' + str(maxValid[1]*100) + '%')
            fig4 = plt.figure()
            fig4 = X_train.groupby([var + '_SPT'])[var + '_SPT'].mean().plot.bar()
            fig4.set_ylabel('moyenne')
            plt.show()

        if(getClasses == True):
            #print('\n --------------' + var + '-------------- \n')
            #print(pd.concat( [X_train.groupby([var + '_SPT'])[var].min(), X_train.groupby([var + '_SPT'])[var].max()], axis=1))
            borneMin = X_train.groupby([var + '_SPT'])[var].min()
            borneMax = X_train.groupby([var + '_SPT'])[var].max()
            classes = pd.DataFrame()
            classes['borneMin'] = borneMin
            classes['borneMax'] = borneMax
            classes.sort_values('borneMin', inplace=True)

            print("'" + str(var) + "': {")
            i=1
            for name in classes.index:
                if (i == 1):
                    print ('\t' + str(i) + ': [0,' + str(classes['borneMax'][name])+ '],')
                elif(i != len(classes)):
                    print ('\t' + str(i) + ': [' + str(classes['borneMin'][name]) + ',' + str(classes['borneMax'][name])+ '],')
                else:
                    print ('\t' + str(i) + ': [' + str(classes['borneMin'][name]) + ',' + str(2*classes['borneMax'][name])+ ']')
                i=i+1
            print('},')
    return dtrRsquared

def linearRegressionCorrelation(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression

    lrRsquared = pd.DataFrame(columns=['LR R²'])
    X_test = X_test.copy(deep=True)
    for var in X_train : 
        
        X = X_train[var].values
        rwrk_Xtrain = X.reshape(-1, 1)
        y = y_train.values
        rwrk_ytrain = y.reshape(-1, 1)

        Xtest = X_test[var].values
        rwrk_Xtest = Xtest.reshape(-1, 1)
        ytest = y_test.values
        rwrk_ytest = ytest.reshape(-1, 1)
        
        
        model = LinearRegression()
        model.fit(rwrk_Xtrain,rwrk_ytrain)
        pred_test = model.predict(rwrk_Xtest)
        X_test[var +'_pred_test'] = pred_test.astype(int)
        X_test[var + '_pred_test'] = X_test[var + '_pred_test'].clip(lower=0)
        
        lrRsquared.loc[var] = round(rsquared(y_test, X_test[var + '_pred_test'])*100,2)
    return lrRsquared

def PolynomialFeaturesCorr(X_train, X_test, y_train, y_test, prOrder):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    prRsquared = pd.DataFrame(columns=['PR' + str(prOrder) + 'R²'])
    X_test = X_test.copy(deep=True)
    pr=PolynomialFeatures(degree=prOrder, include_bias=False)
    model = LinearRegression()
    for var in X_train : 
        
        X_train_pr=pr.fit_transform(X_train[[var]])
        X_test_pr=pr.fit_transform(X_test[[var]])
        
        #X = X_train_pr.values
        #rwrk_Xtrain = X.reshape(-1, 1)
        y = y_train.values
        rwrk_ytrain = y.reshape(-1, 1)

        #Xtest = X_test_pr.values
        #rwrk_Xtest = Xtest.reshape(-1, 1)
        ytest = y_test.values
        rwrk_ytest = ytest.reshape(-1, 1)
        
        model.fit(X_train_pr,rwrk_ytrain)
        pred_test = model.predict(X_test_pr)
        X_test[var +'_pred_test'] = pred_test.astype(int)
        X_test[var + '_pred_test'] = X_test[var + '_pred_test'].clip(lower=0)
        
        prRsquared.loc[var] = round(rsquared(y_test, X_test[var + '_pred_test'])*100,2)
    return prRsquared

def PolynomialRegrTransformationReturnDF(df, varToTransform, prOrder):
    #Cette fonction prends en entrée deux data frame et les renvoi dans une liste après qu'ils aient été transformé

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn_pandas import DataFrameMapper
    from sklearn.preprocessing import StandardScaler

    pr=PolynomialFeatures(degree=prOrder, include_bias=False)
    columnList = []
 

    #Prepare list of columns name for prOrder = 2
    if(prOrder == 2):
        varRestante = varToTransform
        for var in varToTransform :
            columnList.append(str(var))
        for column in df[varToTransform].columns :
            for var in varRestante:
                if (column == var):
                    columnList.append(str(column) + '²')
                else:
                    columnList.append(str(column) + str(var))
            varRestante = np.delete(varRestante, 0)


    #Do the transformation   
    transformed = pr.fit_transform(df[varToTransform])
    transformed_features = pd.DataFrame(transformed, index=df.index)
    
    #Add the name of the columns to the data frame
    j=1

    for col in transformed_features:
        if(prOrder > 2):
            colName = 'PR' + str(prOrder) + '_' + str(j)
            transformed_features.rename(columns={col: colName}, inplace=True)
            columnList.append(colName)
        if(prOrder == 2):
            transformed_features.rename(columns={col: columnList[j-1]}, inplace=True)
        j=j+1


    #Supprime les x premières colonnes qui sont des duplicata
    for i in range(len(varToTransform)):
        transformed_features.drop(transformed_features.columns[0], axis=1, inplace=True)
        del columnList[0]

    #Reset les index et concatène
    df.reset_index(drop=True, inplace=True) 
    transformed_features.reset_index(drop=True, inplace=True) 
    newDfComplete = pd.concat([df, transformed_features], axis=1, sort=True)

    print(columnList)
    return newDfComplete



def replaceByGroupMedian(df, tableOfReplacements, columnContainingNAN, columnUsedForJoin):
    '''
    Cette fonction a été développé pour effectuer des remplacements de valeur NAN dans un dataset
    1 - On regarde le dataset train, on groupe les lignes selon une variable, par exemple neighborhood
	2 - pour chaque modalité de neighoborhood, on récupère la valeure médiane d'une autre variable, par exemple le prix
	3 - ainsi on obtient un df associant à chaque quartier un prix : quartier 1 = 100k, quartier 2 = 150k, quartier 3 = 128k par exemple
	on fait cela de cette façon : X_train_price_medians_by_Neighborhood = X_train.groupby("Neighborhood")["price"].median().to_frame()
	X_train_price_medians_by_Neighborhood est le paramètre tableOfReplacements envoyé à la fonction

	A partir de cette tableOfReplacements, on veut remplacer les valeur NAN dans le dataset df (paramètre de la fonction)
	On regroupe donc les lignes selon la même variable (columnUsedForJoin)
	On effectue le remplacement des NAN dans la colonne concerné (columnContainingNAN)
    '''

    listLignes = df[df[columnContainingNAN].isnull()].index
    for ind in listLignes:
        cellWithNAN = df[columnContainingNAN][ind]
        cellUsedForJoin = df[columnUsedForJoin][ind]
        targetValue = tableOfReplacements.loc[cellUsedForJoin, columnContainingNAN]
        df.loc[ind, columnContainingNAN] = targetValue




def scale_features(dataset, features, scaleMethod) :
    from sklearn import preprocessing
    import copy
    if (features == [] or features == ''):
        features = dataset.columns
    df = copy.deepcopy(dataset)
    
    if(scaleMethod == 0):
        return df
    
    if(scaleMethod == 1):
        for var in df[features] :
                df[var] = (df[var]-df[var].min()) / (df[var].max()-df[var].min())
    
    if(scaleMethod == 2):
        for var in df[features] :
                df[var]=df[var]/df[var].max()
                
    if(scaleMethod == 3):
        for var in df[features] :                
                df[var] = (df[var]-df[var].mean()) / df[var].std()
                
    if(scaleMethod == 4):
        df = preprocessing.StandardScaler().fit(df[features]).transform(df[features])
        df = pd.DataFrame(data=df[0:,0:],    
                      index=dataset.index,    
                      columns=dataset[features].columns)
    
    return df



''' ---------------------- FEATURES SELECTION ---------------------- '''

def RFR_select_features(X, y):
    from sklearn.ensemble import RandomForestRegressor

    feat_names = X.columns
    rf = RandomForestRegressor()
    rf.fit(X, y)

    soluce = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feat_names), 
                 reverse=True)
    new_list = [tuple(i[1] for i in soluce)]
    x = []
    for item in new_list:
        x.extend(item)
    return x, soluce


def ExhaustiveFeatureSelector(X,y, min_features=1 , max_features=4):
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()

    efs1 = EFS(lr, 
               min_features=min_features,
               max_features=max_features,
               scoring='r2',
               print_progress=True,
               cv=5)

    efs1 = efs1.fit(X,y)

    #print('Best subset:', efs1.best_idx_)
    print('Best subset (corresponding names):', efs1.best_feature_names_)
    print('Best R² score: %.2f' % efs1.best_score_ )
    return efs1.best_feature_names_ , efs1.best_score_




def recursive_feature_addition(X_train, y_train, X_test, y_test, model):
    #from sklearn.linear_model import LinearRegression
    # array to hold the feature to be keept.
    colonnes = X_train.columns
    features_to_keep = [colonnes[0]]

    # set this value according to you.
    threshold = 0.0001

    # create your prefered model and  fit it to the training data.
    model_one_feature = model
    model_one_feature.fit(X_train[features_to_keep], y_train)

    # evaluate against your metric.
    y_pred_test = model_one_feature.predict(X_test[features_to_keep])
    score =  rsquared(y_test, y_pred_test)

    # start iterating from the feature.
    for feature in colonnes[1:]:    
        # fit model with  the selected features and the feature to be evaluated
        #model = LinearRegression()
        model.fit(X_train[features_to_keep + [feature]], y_train)
        y_pred_test = model.predict(X_test[features_to_keep + [feature]])
        score_int =  rsquared(y_test, y_pred_test)

        # determine the drop in the roc-auc
        diff_score = score_int - score

        # compare the drop in roc-auc with the threshold
        if diff_score >= threshold:
            
            # if the increase in the roc is bigger than the threshold
            # we keep the feature and re-adjust the roc-auc to the new value
            # considering the added feature
            score = score_int
            features_to_keep.append(feature)

    # print the feature to keep.
    #print(features_to_keep)
    print(score)
    print(len(features_to_keep))
    return features_to_keep, score

def feature_behavior_observation(X_train, y_train, X_test, y_test, model):
    import matplotlib.pyplot as plt
    models2 = {
        'model' : {
            'label' : 'Model',
            'function' : model
        },
    }
    rsquared_list = []
    preds= pd.DataFrame()

    for i in range(1,len(X_train.columns)+1):

        model.fit(X_train[X_train.columns[0:i]], y_train)

        pred = model.predict(X_test[X_test.columns[0:i]])
        preds['temp'] = pred.astype(float)
        preds['pred'] = preds['temp'].clip(lower=0)
        preds.drop(['temp'], axis=1, inplace=True)
        r2 = rsquared(preds['pred'], y_test)

        rsquared_list = rsquared_list + [r2]
    #print(rsquared_list)
    print('max R² = ' + str(round(rsquared_list[rsquared_list.index(max(rsquared_list))],3)))

    print('Index of max R² : ' + str(rsquared_list.index(max(rsquared_list))))
    print(X_train.columns[0:rsquared_list.index(max(rsquared_list))])


    plt.figure(figsize=(10,3))
    plt.plot(range(0, i), rsquared_list)
    plt.fill_between(range(0, i),
                     rsquared_list,
                     alpha=.10)
    plt.legend(('Accuracy legen'))
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.xticks(range(1, i))
    plt.show()


def recursive_feature_elimination_rf(X_train, y_train, X_test, y_test, model, tol=0.0001):
    #https://github.com/Yimeng-Zhang/feature-engineering-and-feature-selection/blob/master/feature_selection/hybrid.py
    from sklearn.linear_model import LinearRegression

    features_to_remove = []
    count = 1
    # initial model using all the features
    model_all_features = model
    model_all_features.fit(X_train, y_train)
    y_pred_test = model_all_features.predict(X_test)
    auc_score_all = rsquared(y_test, y_pred_test)
    
    for feature in X_train.columns:
        #print()
        #print('testing feature: ', feature, ' which is feature ', count,
        #  ' out of ', len(X_train.columns))
        count += 1
        
        # fit model with all variables minus the removed features
        # and the feature to be evaluated
        model.fit(X_train.drop(features_to_remove + [feature], axis=1), y_train)
        y_pred_test = model.predict(
                    X_test.drop(features_to_remove + [feature], axis=1))
        auc_score_int = rsquared(y_test, y_pred_test)
        #print('New Test ROC AUC={}'.format((auc_score_int)))
    
        # print the original roc-auc with all the features
        #print('All features Test ROC AUC={}'.format((auc_score_all)))
    
        # determine the drop in the roc-auc
        diff_auc = auc_score_all - auc_score_int
    
        # compare the drop in roc-auc with the tolerance
        if diff_auc >= tol:
            #print('Drop in ROC AUC={}'.format(diff_auc))
            #print('keep: ', feature)
            score = diff_auc
            
        else:
            #print('Drop in ROC AUC={}'.format(diff_auc))
            #print('remove: ', feature)
            
            # if the drop in the roc is small and we remove the
            # feature, we need to set the new roc to the one based on
            # the remaining features
            auc_score_all = auc_score_int
            score = auc_score_int
            
            # and append the feature to remove to the list
            features_to_remove.append(feature)
    print('score : ' + str(score))
    print('total features to remove: ', len(features_to_remove))  
    features_to_keep = [x for x in X_train.columns if x not in features_to_remove]
    print('total features to keep: ', len(features_to_keep))
    
    return features_to_keep, score

    