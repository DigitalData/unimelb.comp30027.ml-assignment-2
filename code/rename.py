# Generate the parameters for the pipeline
parameters = [
    { # Multinomial Naive Bayes
        'clf': [MultinomialNB()],

        'clf__alpha': [0, 1, 10],

        'clf__fit_prior': [True, False],

    },
    { # Bernoulli Naive Bayes
        'clf': [BernoulliNB()],

        'clf__alpha': [0, 1, 10],

        'clf__fit_prior': [True, False],

    },
    { # Logistic Regressions
        'clf': [LogisticRegression()],

        'clf__max_iter': [100, 500],

        'clf__fit_intercept': [True, False],

        'clf__solver': ['sag', 'saga', 'lbfgs'],
    },
    { # Decision Trees
        'clf': [DecisionTreeClassifier()],

        'clf__max_depth': [None, 1, 100, 1000],

    },
    { # K Nearest Neighbours
        'clf': [KNeighborsClassifier()],

        'clf__n_neighbors': [1, 5, 100, 1000],

    },
    { # Support Vector Classifiers
        'clf': [SVC()],

        'clf__kernel': ['poly'],

        'clf__degree': [3, 10],

        'clf__C': [0.01, 0.5, 1, 3],
    },
    { # Linear Support Vector Classifiers
        'clf': [LinearSVC()],

        'clf__C': [0.01, 0.5, 1, 3]
    },
]