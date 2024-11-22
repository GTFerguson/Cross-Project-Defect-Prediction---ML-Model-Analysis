import weka.classifiers.Evaluation;

// Weka's Evaluation class does not allow for storage of the training and test dataset names, so we must create our
// own custom class for this purpose.
class EvaluationResult {
    // Datasets
    private String training_set_name;
    private String testing_set_name;
    // Feature Selection
    private String evaluator;
    private String search_method;
    // Results of test
    private Evaluation evaluation;

    public EvaluationResult (String training_set_name, String testing_set_name, Evaluation evaluation) {
        this.training_set_name  = training_set_name;
        this.testing_set_name   = testing_set_name;
        this.evaluator          = "None";
        this.search_method      = "None";
        this.evaluation         = evaluation;
    }

    public EvaluationResult (String training_set_name, String testing_set_name,
                             Evaluation evaluation,
                             String evaluator, String search) {
        this.training_set_name  = training_set_name;
        this.testing_set_name   = testing_set_name;
        this.evaluator          = evaluator;
        this.search_method      = search;
        this.evaluation         = evaluation;
    }

    public String       get_training_set_name ()    { return training_set_name; }
    public String       get_evaluator ()            { return evaluator;         }
    public String       get_search_method ()        { return search_method;     }
    public String       get_testing_set_name ()     { return testing_set_name;  }
    public Evaluation   get_evaluation ()           { return evaluation;        }
}