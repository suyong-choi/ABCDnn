import numpy as np

class OneHotEncoder_int(object):
    """One hot encoder for integer inputs with overflows
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self, categorical_features, lowerlimit=None, upperlimit=None):
        self.iscategorical = categorical_features
        self.ncolumns = len(categorical_features)
        self.ncats=0
        self.categories_per_feature = []

        self.ncatgroups = 0
        for b in categorical_features:
            if b:
                self.ncatgroups += 1
        self.lowerlimit = lowerlimit # initial set to the input, but will be checked later
        self.upperlimit = upperlimit # initial set to the input, but will be checked later
        self.categories_fixed = False
        pass

    def applylimit(self, categoricalinputdata):
        # should check whether lower limit set makes sense
        if self.lowerlimit is None:
            self.lowerlimit = np.min(categoricalinputdata, axis=0)
        else:
            self.lowerlimit = np.maximum(self.lowerlimit, np.min(categoricalinputdata, axis=0))

        # should check whether upper limit set makes sense
        if self.upperlimit is None:
            self.upperlimit = np.max(categoricalinputdata, axis=0)
        else:
            self.upperlimit = np.minimum(self.upperlimit, np.max(categoricalinputdata, axis=0))

        lowerlimitapp = np.maximum(categoricalinputdata, self.lowerlimit)
        #limitapp = np.minimum(lowerlimitapp, self.upperlimit).astype(int)
        limitapp = np.minimum(lowerlimitapp, self.upperlimit)
        return limitapp

    def _encode(self, inputdata):
        categorical_columns=inputdata[:, self.iscategorical]
        float_columns=inputdata[:, [not i for i in self.iscategorical]]

        cat_limited = self.applylimit(categorical_columns)-self.lowerlimit.astype(int)

        catshape = categorical_columns.shape

        arraylist=[]
        if not self.categories_fixed:
            for cat in range(catshape[1]):
                ncats = int(self.upperlimit[cat] - self.lowerlimit[cat] + 1) # number of categories
                self.categories_per_feature.append(ncats)
                self.ncats += ncats
            self.categories_fixed = True

        for cat in range(catshape[1]):
            ncats = int(self.upperlimit[cat] - self.lowerlimit[cat] + 1) # number of categories
            res = np.eye(ncats)[cat_limited[:,cat]]
            #print(res)
            arraylist.append(res)
        if float_columns.shape[1]>0:
            arraylist.append(float_columns)
        encoded = np.concatenate(tuple(arraylist), axis=1).astype(np.float32)
        return encoded

    def encode(self, inputdata):

        cat_limited = self.applylimit(inputdata)-self.lowerlimit

        # one hot encoding information
        if not self.categories_fixed:
            for icol, iscat in zip(range(self.ncolumns), self.iscategorical):
                if iscat:
                    ncats = int(self.upperlimit[icol] - self.lowerlimit[icol] + 1) # number of categories
                    self.categories_per_feature.append(ncats)
                    self.ncats += ncats
                else:
                    self.categories_per_feature.append(0)
            self.categories_fixed = True

        # the actual encoding part
        arraylist=[]
        for icol, ncat_feat in zip(range(self.ncolumns), self.categories_per_feature):
            if ncat_feat>0:
                res = np.eye(ncat_feat)[cat_limited[:,icol].astype(int)]
                arraylist.append(res)
            else:
                arraylist.append(inputdata[:,icol].reshape((inputdata.shape[0], 1)))

        encoded = np.concatenate(tuple(arraylist), axis=1).astype(np.float32)
        return encoded
    
    def encodedcategories(self):
        return self.ncats

    def transform(self, inputdata):
        return self.encode(inputdata)

    def _decode(self, onehotdata):
        colstart = 0
        
        arraylist = []
        for i in range(self.ncatgroups):
            ncats = int(self.upperlimit[i] - self.lowerlimit[i]+1)  # number of categories
            datatoconvert = onehotdata[:, colstart:colstart+ncats]
            converted = np.argmax(datatoconvert, axis=1) + self.lowerlimit[i]
            converted = np.reshape(converted, newshape=(converted.shape[0], 1))
            arraylist.append(converted)
            colstart += ncats
        if colstart<onehotdata.shape[1]:
            arraylist.append(onehotdata[:, colstart:])
        decoded = np.concatenate(tuple(arraylist), axis=1)
        return decoded

    def decode(self, onehotdata):
        current_col = 0 # start from column 0
        arraylist = []
        for ifeat, ncats in zip(range(len(self.categories_per_feature)), self.categories_per_feature):
            if ncats>0:
                datatoconvert = onehotdata[:, current_col:current_col+ncats]
                converted = np.argmax(datatoconvert, axis=1) + self.lowerlimit[ifeat]
                converted = np.reshape(converted, newshape=(converted.shape[0], 1))
                arraylist.append(converted)
                current_col += ncats
            else:
                arraylist.append(onehotdata[:, current_col].reshape((onehotdata.shape[0], 1)))
                current_col += 1
        decoded = np.concatenate(tuple(arraylist), axis=1)
        return decoded

    pass


def test():
    x = np.array([[ 0,  1,  2], [ 3,  4,  5], [ 6,  7,  8], [ 9, 10, 11]])
    ohe = OneHotEncoder_int(categorical_features=[True, False, True], lowerlimit=[2,0,2], upperlimit=[8,100,8])
    xlimited = ohe.applylimit(x)
    print(xlimited)
    encodedx = ohe.encode(x)
    print(encodedx)
    decoded = ohe.decode(encodedx)
    print(decoded)
    print()
    ohe2 = OneHotEncoder_int(categorical_features=[True, False,True ])
    encodedx = ohe2.encode(x)
    decoded = ohe2.decode(encodedx)
    print(decoded)
    pass


if __name__ == "__main__":
    test()