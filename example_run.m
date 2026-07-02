function example_run()
% -------------------------------------------------------------------------
% 1. Setup and Initialization
% -------------------------------------------------------------------------
clear;              
clc;                  
close all;            
addpath('common', 'classifiers', 'evaluation'); 

% -------------------------------------------------------------------------
% 2. Configuration Parameters
% -------------------------------------------------------------------------
DATASET_NAME = 'Flags';       
L_original = 7;              
k_knn = 10;                    

delta1 = 0.1;               
delta2 = 0.1;                 
k1 = 0.7;                      
k2 = 1.0;                       

disp('--- Configuration ---');
disp(['Delta1: ', num2str(delta1), ', Delta2: ', num2str(delta2), ...
      ', k1: ', num2str(k1), ', k2: ', num2str(k2), ', k-NN: ', num2str(k_knn)]);
disp('------------------------------');

% -------------------------------------------------------------------------
% 3. Data Loading and Initial Partition
% -------------------------------------------------------------------------
disp('1. Loading training and testing data...');
train_dataset = load(['data/', DATASET_NAME, '/', DATASET_NAME, '-train.mat']);
test_dataset = load(['data/', DATASET_NAME, '/', DATASET_NAME, '-test.mat']);
train_data_raw = double(train_dataset.train); 
test_data_raw = double(test_dataset.test);    
disp('Data loaded.');

disp('2. Partitioning data and setting labels...');
[~, p_raw] = size(train_data_raw);
num_features = p_raw - L_original;
X_train = train_data_raw(:, 1:num_features);
Y_train_raw = train_data_raw(:, (num_features + 1):end);
X_test = test_data_raw(:, 1:num_features);
Y_test_raw = test_data_raw(:, (num_features + 1):end);

Y_train_binarized = Y_train_raw; 
Y_test_binarized = Y_test_raw;
L_new = size(Y_train_binarized, 2);
disp(['Labels kept at L_original: ', num2str(L_original), ...
      '. New binary labels: ', num2str(L_new), ' (DIMENSION PRESERVED)']); 

% -------------------------------------------------------------------------
% 4. Data Encoding
% -------------------------------------------------------------------------
disp('3. Encoding features and labels to 1-based discrete values (Unified Rule)...');
num_cols_X = size(X_train, 2);              
num_cols_Y = size(Y_train_binarized, 2);     
X_encoding_map = cell(1, num_cols_X);       
Y_encoding_map = cell(1, num_cols_Y);       
X_train_encoded = zeros(size(X_train), 'double');          
X_test_encoded = zeros(size(X_test), 'double');            
Y_train_1based = zeros(size(Y_train_binarized), 'double'); 
Y_test_1based = zeros(size(Y_test_binarized), 'double');  

for j = 1:num_cols_X
    [X_train_encoded(:, j), X_encoding_map{j}] = encode_discrete_1based(X_train(:, j));
    X_test_encoded(:, j) = map_to_1based(X_test(:, j), X_encoding_map{j});
end

for j = 1:num_cols_Y
    [Y_train_1based(:, j), Y_encoding_map{j}] = encode_discrete_1based(Y_train_raw(:, j));
    Y_test_1based(:, j) = map_to_1based(Y_test_raw(:, j), Y_encoding_map{j});
end

train_data_processed = [X_train_encoded, Y_train_1based];

trainNew = train_data_processed;
train_save_path = ['./data/', DATASET_NAME, '/', DATASET_NAME, '-trainNew_La.mat'];
fprintf('   Saving encoded training data to: %s\n', train_save_path);
save(train_save_path, 'trainNew'); 

testNew = [X_test_encoded, Y_test_1based]; 
test_save_path = ['./data/', DATASET_NAME, '/', DATASET_NAME, '-testNew_La.mat'];
fprintf('   Saving encoded testing data to: %s\n', test_save_path);
save(test_save_path, 'testNew');

% -------------------------------------------------------------------------
% 5. SCAMFS Feature Selection
% -------------------------------------------------------------------------
disp('4. Running SCAMFS algorithm on processed data...');

fs_timer = tic;
selected_features = SCAMFS(train_data_processed, L_new, delta1, delta2, k1, k2); 
fs_elapsed_time = toc(fs_timer);
num_selected_features = length(selected_features);

disp(['SCAMFS completed. ', num2str(num_selected_features), ' features selected.']);
fprintf('SCAMFS core selection time: %.2f seconds\n', fs_elapsed_time);

% -------------------------------------------------------------------------
% 6. ML-kNN
% -------------------------------------------------------------------------
disp('5. Training ML-kNN and predicting on the test set...');

X_train_final = X_train_encoded(:, selected_features);
X_test_final = X_test_encoded(:, selected_features);   

[Y_pred, F_pred] = ML_KNN(X_train_final, Y_train_binarized, X_test_final, k_knn);
disp('Prediction completed.');

% -------------------------------------------------------------------------
% 7.Evaluation and Display
% -------------------------------------------------------------------------
disp('6. Calculating all performance metrics...');
ham_loss = calculate_hamming_loss(Y_test_binarized, Y_pred);
sub_acc = calculate_subset_accuracy(Y_test_binarized, Y_pred);

[macro_f1, micro_f1] = calculate_f_scores(Y_test_binarized, Y_pred);

[avg_prec, coverage, rank_loss] = calculate_ranking_metrics(Y_test_binarized, F_pred);

disp(' ');
disp('==================== FINAL RESULTS ====================');
MetricNames = ["Hamming Loss (↓)"; "Subset Accuracy (↑)"; "Average Precision (↑)"; ...
               "Coverage (↓)"; "Ranking Loss (↓)"; "Macro-F1 (↑)"; "Micro-F1 (↑)"];
Values = [ham_loss; sub_acc; avg_prec; coverage; rank_loss; macro_f1; micro_f1];
results = table(MetricNames, Values);
disp(results);             
disp('==============================================================');

% -------------------------------------------------------------------------
% 8. Final Data Saving and Output
% -------------------------------------------------------------------------
fprintf('\n');
disp('7. Saving the final five datasets used for ML-kNN ...'); 

FinalXTrain = X_train_final;
save(['./data/', DATASET_NAME, '/FinalXTrain_La.mat'], 'FinalXTrain');

FinalYTrain = Y_train_binarized;
save(['./data/', DATASET_NAME, '/FinalYTrain_La.mat'], 'FinalYTrain');

FinalXTest = X_test_final;
save(['./data/', DATASET_NAME, '/FinalXTest_La.mat'], 'FinalXTest');

FinalYTest = Y_test_binarized;
save(['./data/', DATASET_NAME, '/FinalYTest_La.mat'], 'FinalYTest');

FinalYPred = Y_pred;
save(['./data/', DATASET_NAME, '/FinalYPred_La.mat'], 'FinalYPred');

disp(['   Five final datasets saved successfully in ./data/', DATASET_NAME, '/.']); 

fprintf('\n');
disp('==================== VARIABLE OUTPUT ====================');
disp('The following variables are available in the workspace and saved to disk:');
disp(['- Training Features: FinalXTrain, Size: ', mat2str(size(FinalXTrain))]);
disp(['- Training Labels: FinalYTrain, Size: ', mat2str(size(FinalYTrain))]);
disp(['- Testing Features: FinalXTest, Size: ', mat2str(size(FinalXTest))]);
disp(['- Testing Labels: FinalYTest, Size: ', mat2str(size(FinalYTest))]);
disp(['- Predicted Labels: FinalYPred, Size: ', mat2str(size(FinalYPred))]); 
disp('=========================================================');


fprintf('\n==================== EXECUTION TIME ====================\n');
fprintf('SCAMFS core selection time: %.2f seconds (approx. %.2f minutes)\n', ...
        fs_elapsed_time, fs_elapsed_time/60);
disp('=========================================================');
end
