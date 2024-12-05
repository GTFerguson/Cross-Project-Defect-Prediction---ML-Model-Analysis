import org.w3c.dom.Attr;
import weka.attributeSelection.*;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

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

    public Map<String, ASEvaluation>    get_evaluators ()                           { return evaluators;                        }
    public ASEvaluation                 get_evaluator (String evaluator)            { return evaluators.get(evaluator);         }
    public Map<String, ASSearch>        get_search_methods ()                       { return search_methods;                    }
    public ASSearch                     get_search_method (String search_method)    { return search_methods.get(search_method); }

    // Returns a feature selector that has been trained on a given dataset
    public AttributeSelection train_selector (
            String evaluator, String search_method, Instances training_set, Double threshold) throws Exception {
        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(this.get_evaluator(evaluator));

        // Threshold is only used on Ranker search so ensure it is the right method
        if (Objects.equals(search_method, "Ranker")) {
            Ranker ranker = new Ranker();
            ranker.setThreshold(threshold);
            selector.setSearch(ranker);
            System.out.printf("Selector threshold set to %.2f%n", threshold);
        } else {
            selector.setSearch(this.get_search_method(search_method));
        }

        selector.SelectAttributes(training_set);
        System.out.println("Original No. of Attributes: " + training_set.numAttributes());
        System.out.println("Reduced No. of Attributes:  " + selector.selectedAttributes().length);
        System.out.println("Selected Attributes:  " + Arrays.toString(selector.selectedAttributes()));
        return selector;
    }

    // Convenience method without threshold
    public AttributeSelection train_selector (String evaluator, String search_method,
                                              Instances training_set) throws Exception {
        return train_selector(evaluator, search_method, training_set, null);
    }
}