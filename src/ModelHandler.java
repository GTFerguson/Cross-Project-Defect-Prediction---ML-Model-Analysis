import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.immune.immunos.*;
import weka.classifiers.trees.J48;

import java.util.HashMap;

public class ModelHandler {
    private HashMap<String, AbstractClassifier> model_map = new HashMap<String, AbstractClassifier>();

    public void load_model (String model_name, Class<? extends AbstractClassifier> model_class) {
        try {
            // Instantiate the model dynamically using reflection
            AbstractClassifier model = model_class.getDeclaredConstructor().newInstance();
            model_map.put(model_name, model);
            System.out.println(model_name + " successfully loaded!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error loading model: " + model_name);
        }
    }

    public void load_models () {
        load_model("Immunos-1",     Immunos1.class);
        load_model("Immunos-2",     Immunos2.class);
        load_model("Immunos-99",    Immunos99.class);
        load_model("J48",           J48.class);
        load_model("Naive Bayes",   NaiveBayes.class);
    }

    public AbstractClassifier get_model (String model_name) { return model_map.get(model_name); }
    public HashMap<String, AbstractClassifier> get_model_map () { return model_map; }
}
