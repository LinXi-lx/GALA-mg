result=./result.txt

for file in ../data/*.bin;  do
    for i in {1..3}; do
        for num in {1,2,3,4}; do
            timeout --foreground 1h mpirun -np $num ./gala_main -f $file -t 0.8 > output.txt
            #gpu/patternmatchgpu -f /data/"$file".txt.bin -d 0 -a 6 -q "$q_value" -p 1 -t 1 -v 1 -m 1 > output.txt
            # $filename=$(echo $file | cut -d . -f1)
            if [ $? -eq 124 ]; then
                    echo "Timeout occurred for file: $file, pattern: $pattern" >> $result
            else
                    movetime=$(grep -m 1 -o -E 'decideandmove time = [^ ]+' output.txt  | cut -d ':' -f2)
                    modtime=$(grep -m 1 -o -E 'weight updating time = [^ ]+' output.txt  | cut -d ':' -f2)
                    remainingtime=$(grep -m 1 -o -E 'remaining time = [^ ]+' output.txt  | cut -d ':' -f2)
                    commtime=$(grep -m 1 -o -E 'comm time = [^ ]+' output.txt  | cut -d ':' -f2)
                    first_louvain=$(grep -m 1 -o -E 'time without data init = [^,]+' output.txt  | cut -d '=' -f2)
                    
                    echo -n "$file:$num" >> $result
                    echo -n " $movetime" >> $result
                    echo -n " $modtime" >> $result

                    echo -n " $remainingtime" >> $result

                    echo -n " $commtime" >> $result
                    echo  " $first_louvain" >> $result


            fi
        
        done
    done
    echo  >> $result
done