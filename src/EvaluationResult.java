import weka.classifiers.Evaluation;

// Weka's Evaluation class does not allow for storage of the training and test dataset names, so we must create our
// own custom class for this purpose.
class EvaluationResult {
    private String training_set_name;
    private String testing_set_name;
    private Evaluation evaluation;

    public EvaluationResult (String training_set_name, String testing_set_name, Evaluation evaluation) {
        this.training_set_name  = training_set_name;
        this.testing_set_name   = testing_set_name;
        this.evaluation         = evaluation;
    }

    public String       get_training_set_name ()    { return training_set_name; }
    public String       get_testing_set_name ()     { return testing_set_name;  }
    public Evaluation   get_evaluation ()           { return evaluation;        }
}