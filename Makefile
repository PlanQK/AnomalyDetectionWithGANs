
# This Makefile is rather dirty and hardcode, but it works
# Feel free to change it and make it more general
parallel:
	./execute_test_script.sh 1 & ./execute_test_script.sh 2 & ./execute_test_script.sh 3 & ./execute_test_script.sh 4 & ./execute_test_script.sh 5 & ./execute_test_script.sh 6 & ./execute_test_script.sh 7 & ./execute_test_script.sh 8 & ./execute_test_script.sh 9 & ./execute_test_script.sh 10 & ./execute_test_script.sh 11 & ./execute_test_script.sh 12 & ./execute_test_script.sh 13 & ./execute_test_script.sh 14 & ./execute_test_script.sh 15 && sudo shutdown

non_parallel:
	python3 test_gan_classifier && sudo shutdown