import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.immune.airs.AIRS2Parallel;
import weka.classifiers.immune.immunos.*;
import weka.classifiers.trees.J48;

import java.util.HashMap;

public class ModelHandler {
    private HashMap<String, AbstractClassifier> model_map = new HashMap<String, AbstractClassifier>();

    public ModelHandler () {
        this.load_model("Immunos-1",        Immunos1.class);
//        this.load_model("Immunos-2",        Immunos2.class);
//        this.load_model("Immunos-99",       Immunos99.class);
//        this.load_model("J48",              J48.class);
//        this.load_model("Naive Bayes",      NaiveBayes.class);
//        this.load_model("AIRS2 Parallel",   AIRS2Parallel.class);
    }

    public void load_model (String model_name, Class<? extends AbstractClassifier> model_class) {
        try {
            // Instantiate the model dynamically using reflection
            AbstractClassifier model = model_class.getDeclaredConstructor().newInstance();
            this.model_map.put(model_name, model);
            System.out.println(model_name + " successfully loaded!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error loading model: " + model_name);
        }
    }

    public AbstractClassifier get_model (String model_name) { return this.model_map.get(model_name); }
    public HashMap<String, AbstractClassifier> get_model_map () { return this.model_map; }
}
