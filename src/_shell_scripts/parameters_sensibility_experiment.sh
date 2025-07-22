ks=(250 500 1000 2000 4000)
epochs=(3 5 10 15 20)
discounts=(0.5 1 2 3 4)
boot_epochs=(3 7 15 25 31 37)

dataset_name="CIFAR10"
seed=42
best_k=1000
best_epochs=10
best_boot_epochs=20
best_discount=2


source /home/scala/.virtualenvs/DataFiltering/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/RePlayItStraight/
cd ../../

echo "Dataset selected ${dataset_name}!"
echo "Running Experiment for sensibility over k..."
for i in "${!ks[@]}"; do
  current_k="${ks[$i]}"
  path_log_file="/home/scala/projects/RePlayItStraight/results/res_${dataset_name}_seed_${seed}_k_${current_k}_epochs_${best_epochs}_discount_${best_discount}_bootep_${best_boot_epochs}.log"
  echo "Running experiment with k: ${current_k}, epochs: ${best_epochs}, boot epochs: ${best_boot_epochs}, discount: ${best_discount}..."
  python src/re_play_it_straight/main_re_play_it_straight.py --gpu 0 --data_path=/home/scala/projects/DataFiltering/dataset --dataset "$dataset_name" --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query "$current_k" --epochs "$best_epochs" --batch-size 128 --n_split 3 --cycle 20 --boot_epochs "$best_boot_epochs" --discount_rs2 "$best_discount" --boost_threshold 0.07 --seed "$seed" > "${path_log_file}" 2>&1
done

echo "Running Experiment for sensibility over epochs..."
for i in "${!epochs[@]}"; do
  current_epochs="${epochs[$i]}"
  path_log_file="/home/scala/projects/RePlayItStraight/results/res_${dataset_name}_seed_${seed}_k_${best_k}_epochs_${current_epochs}_discount_${best_discount}_bootep_${best_boot_epochs}.log"
  echo "Running experiment with k: ${best_k}, epochs: ${best_epochs}, boot epochs: ${best_boot_epochs}, discount: ${best_discount}..."
  python src/re_play_it_straight/main_re_play_it_straight.py --gpu 0 --data_path=/home/scala/projects/DataFiltering/dataset --dataset "$dataset_name" --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query "$best_k" --epochs "$current_epochs" --batch-size 128 --n_split 3 --cycle 20 --boot_epochs "$best_boot_epochs" --discount_rs2 "$best_discount" --boost_threshold 0.07 --seed "$seed" > "${path_log_file}" 2>&1
done

echo "Running Experiment for sensibility over discounts..."
for i in "${!discounts[@]}"; do
  current_discount="${discounts[$i]}"
  path_log_file="/home/scala/projects/RePlayItStraight/results/res_${dataset_name}_seed_${seed}_k_${best_k}_epochs_${best_epochs}_discount_${current_discount}_bootep_${best_boot_epochs}.log"
  echo "Running experiment with k: ${best_k}, epochs: ${best_epochs}, boot epochs: ${best_boot_epochs}, discount: ${best_discount}..."
  python src/re_play_it_straight/main_re_play_it_straight.py --gpu 0 --data_path=/home/scala/projects/DataFiltering/dataset --dataset "$dataset_name" --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query "$best_k" --epochs "$best_epochs" --batch-size 128 --n_split 3 --cycle 20 --boot_epochs "$best_boot_epochs" --discount_rs2 "$current_discount" --boost_threshold 0.07 --seed "$seed" > "${path_log_file}" 2>&1
done

echo "Running Experiment for sensibility over boot epochs..."
for i in "${!boot_epochs[@]}"; do
  current_boot_epochs="${boot_epochs[$i]}"
  path_log_file="/home/scala/projects/RePlayItStraight/results/res_${dataset_name}_seed_${seed}_k_${best_k}_epochs_${best_epochs}_discount_${current_discount}_bootep_${current_boot_epochs}.log"
  echo "Running experiment with k: ${best_k}, epochs: ${best_epochs}, boot epochs: ${current_boot_epochs}, discount: ${best_discount}..."
  python src/re_play_it_straight/main_re_play_it_straight.py --gpu 0 --data_path=/home/scala/projects/DataFiltering/dataset --dataset "$dataset_name" --n-class 10 --model ResNet18 --method ImprovedUncertainty --uncertainty LeastConfidence --n-query "$best_k" --epochs "$best_epochs" --batch-size 128 --n_split 3 --cycle 20 --boot_epochs "$current_boot_epochs" --discount_rs2 "$best_discount" --boost_threshold 0.07 --seed "$seed" > "${path_log_file}" 2>&1
done

echo "Completed!"
