import weka.attributeSelection.*;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.HashMap;
import java.util.Map;

public class FeatureSelection {
    private Map<String, ASEvaluation> evaluators = new HashMap<>();
    private Map<String, ASSearch> search_methods = new HashMap<>();

    public Map<String, ASEvaluation> (String evaluator) {
        return this.evaluators.get(evaluator);
    }
    public Map<String, ASSearch> get_search_method (String search_method) {
        return this.search_methods.get(search_method);
    }

    // Applies the specified filter to the given dataset, returning the filtered dataset
    public Instances apply_feature_selection(Instances dataset, ASEvaluation evaluator, ASSearch search) {
        AttributeSelection filter = new AttributeSelection();
        filter.setEvaluator(evaluator);
        filter.setSearch(search);
        return Filter.useFilter(dataset, filter);
    }

    public FeatureSelection() {
        this.evaluators.put("CFS Subset", new CfsSubsetEval());
        this.evaluators.put("Info Gain", new InfoGainAttributeEval());
        this.evaluators.put("Gain Ratio", new GainRatioAttributeEval());
        this.search_methods.put("Best First", new BestFirst());
        this.search_methods.put("Greedy Stepwise", new GreedyStepwise());
        this.search_methods.put("Ranker", new Ranker());
    }

}
