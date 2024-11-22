import weka.attributeSelection.*;
import weka.core.Instances;
import weka.filters.Filter;
import java.util.HashMap;
import java.util.Map;

public class FeatureSelection {
    private Map<String, ASEvaluation> evaluators = new HashMap<>();
    private Map<String, ASSearch> search_methods = new HashMap<>();

    public FeatureSelection () {
        this.evaluators.put("CFS Subset", new CfsSubsetEval());
        this.evaluators.put("Info Gain", new InfoGainAttributeEval());
        this.evaluators.put("Gain Ratio", new GainRatioAttributeEval());
        this.search_methods.put("Best First", new BestFirst());
        this.search_methods.put("Greedy Stepwise", new GreedyStepwise());
        this.search_methods.put("Ranker", new Ranker());
    }

    public Map<String, ASEvaluation> get_evaluators() {
        return evaluators;
    }

    public ASEvaluation get_evaluator (String evaluator) {
        return this.evaluators.get(evaluator);
    }

    public Map<String, ASSearch> get_search_methods() {
        return search_methods;
    }

    public ASSearch get_search_method (String search_method) {
        return this.search_methods.get(search_method);
    }

    // Returns a feature selector that has been trained on a given dataset
    public AttributeSelection train_selector (String evaluator, String search_method, Instances training_set) throws Exception {
        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(this.get_evaluator(evaluator));
        selector.setSearch(this.get_search_method(search_method));
        selector.SelectAttributes(training_set);
        return selector;
    }

    // Applies the specified filter to the given dataset, returning the filtered dataset
    public Instances apply_feature_selection(Instances dataset, String evaluator, String search) throws Exception {
        AttributeSelection selector = this.train_selector(evaluator, search, dataset);
        // Reduce dimensionality of the dataset
        return selector.reduceDimensionality(dataset);
    }
}