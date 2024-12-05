import weka.classifiers.Evaluation;

// Weka's Evaluation class does not allow for storage of the training and test dataset names, so we must create our
// own custom class for this purpose.
class EvaluationResult {
    // Model and Dataset names
    private String model_name;
    private String training_set_name;
    private String testing_set_name;
    // Feature Selection details
    private String evaluator;
    private String search_method;
    private double threshold;
    // Results of test
    private Evaluation evaluation;

    public EvaluationResult (String model_name, String training_set_name, String testing_set_name,
                             String evaluator, String search_method, double threshold,
                             Evaluation evaluation) {
        this.model_name         = model_name;
        this.training_set_name  = training_set_name;
        this.testing_set_name   = testing_set_name;
        this.evaluator          = evaluator;
        this.search_method      = search_method;
        this.threshold          = threshold;
        this.evaluation         = evaluation;
    }

    // Convenience factory method for no feature selection
    public static EvaluationResult create(String model_name, String training_set_name, String testing_set_name,
                                          Evaluation evaluation) {
        return new EvaluationResult
                (model_name, training_set_name, testing_set_name, null, null, 0.0, evaluation);
    }

    // Convenience factory method for feature selection without threshold
    public static EvaluationResult create(String model_name, String training_set_name, String testing_set_name,
                                          String evaluator, String search_method, Evaluation evaluation) {
        return new EvaluationResult
                (model_name, training_set_name, testing_set_name, evaluator, search_method, 0.0, evaluation);
    }

    // Helper method to help ensure a safe value is returned from metric getters
    private double safe_metric_value (Double value) {
        return Double.isNaN(value) ? 0.0 : value;
    }

    // GETTERS
    public String       get_model_name ()           { return model_name;        }
    public String       get_training_set_name ()    { return training_set_name; }
    public String       get_evaluator ()            { return evaluator;         }
    public String       get_search_method ()        { return search_method;     }
    public double       get_threshold ()            { return threshold;         }
    public String       get_testing_set_name ()     { return testing_set_name;  }
    public Evaluation   get_evaluation ()           { return evaluation;        }

    // These getters get data from the Evaluation object
    public double get_accuracy () {
        // Divide by 100 to get as percentage
        return safe_metric_value(evaluation.pctCorrect()/100);
    }

    public double get_recall () {
        return safe_metric_value(evaluation.recall(1));
    }

    public double get_f_measure () {
        return safe_metric_value(evaluation.fMeasure(1));
    }

    // SETTERS
    public void set_model_name (String model_name)          { this.model_name           = model_name;       }
    public void set_evaluator (String evaluator)            { this.evaluator            = evaluator;        }
    public void set_search_method (String search_method)    { this.search_method        = search_method;    }
    public void set_threshold (double threshold)            { this.threshold            = threshold;        }
}