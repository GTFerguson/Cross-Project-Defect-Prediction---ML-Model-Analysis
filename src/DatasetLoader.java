import weka.attributeSelection.AttributeSelection;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.HashMap;
import java.util.Map;

public class DatasetLoader {
    private HashMap<String, Instances> datasets = new HashMap<String, Instances>();
    private HashMap<String, Map<String, Instances>> coral_datasets = new HashMap<>();

    public DatasetLoader () {
        load_dataset("CM1", "datasets/promise/cm1.arff");
        load_dataset("JM1", "datasets/promise/jm1.arff");
        load_dataset("KC1", "datasets/promise/kc1.arff");
        load_dataset("KC2", "datasets/promise/kc2.arff");
        load_dataset("PC1", "datasets/promise/pc1.arff");
    }

    // Load and store a dataset into the datasets map
    public void load_dataset (String dataset_name, String filepath) {
        try {
            DataSource source = new DataSource(filepath);
            Instances dataset = source.getDataSet();

            // Set the class index attribute if it is not provided
            if (dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes()- 1);

            datasets.put(dataset_name, dataset);
            System.out.println(dataset_name + " successfully loaded!");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void load_aligned_datasets(String[] filepaths) throws Exception {
        for (String filepath : filepaths) {
            // Get the filename and remove the extension and convert to uppercase
            String filename = filepath.substring(filepath.lastIndexOf("/") + 1).replace(".arff", "").toUpperCase();
            // ... Then we split the string, first part being the source, second part being the target.
            String[] parts = filename.split("-");

            // Ensure the filename provided is the expected format
            if (parts.length != 2) {
                System.err.println("Invalid file format: " + filepath);
                continue;
            }

            // Ensure source and target datasets exist
            if (!datasets.containsKey(parts[0]) || !datasets.containsKey(parts[1])) {
                System.err.printf("Skipping %s. Source (%s) or target (%s) dataset not loaded.%n",
                    filepath, parts[0], parts[1]
                );
                continue;
            }

            DataSource source = new DataSource(filepath);
            Instances dataset = source.getDataSet();

            // Set the class index attribute if it is not provided
            if (dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes()- 1);

            // Insert the adjusted dataset...
            // Inner Key: source dataset name
            // Outer Key: target dataset name
            coral_datasets.computeIfAbsent(parts[0], k -> new HashMap<>()).put(parts[1], dataset);

            System.out.println(filename + " successfully loaded!");

        }
    }

    public Instances get_dataset (String dataset_name) {
        return datasets.get(dataset_name);
    }

    public Map<String, Instances> get_coral_dataset (String dataset_name) {
        return coral_datasets.get(dataset_name);
    }

    public HashMap<String, Instances> get_datasets() { return datasets; }
    public HashMap<String, Map<String, Instances>> get_coral_datasets() { return coral_datasets; }

    // Preprocess a single dataset (apply ReplaceMissingValues filter)
    private Instances preprocess(Instances data) {
        try {
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            return Filter.useFilter(data, replaceMissing);
        } catch (Exception e) {
            e.printStackTrace();
            return data;  // Return original data if preprocessing fails
        }
    }

    // Preprocess all datasets in the HashMap
    public void preprocess_datasets() {
        for (Map.Entry<String, Instances> entry : datasets.entrySet()) {
            Instances processedData = preprocess(entry.getValue());
            datasets.put(entry.getKey(), processedData);
        }
        System.out.println("All datasets preprocessed successfully!");
    }

    public void set_dataset (String key, Instances dataset) {
        datasets.put(key, dataset);
    }
}