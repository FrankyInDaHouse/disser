if exist('csv_table.csv', 'file')
    clear;
    fclose('all');
    delete('csv_table.csv');
end
textHeader = "received_seq, is_cw";
fid = fopen('csv_table.csv','w');
fprintf(fid,'%s\n',textHeader);

% Parameters of code
l = 15;
amount_of_seq = 2^l;
stat_ones = 0;
stat_zeros = 0;
for iterator = 1 : amount_of_seq
    value = iterator - 1;
    data_vector = fliplr( de2bi(value, l) );
    ones = sum(data_vector == 1);
    zeros = sum(data_vector == 0);
    
    if (sum(data_vector == 1) > sum(data_vector == 0))
        answer = 1;
        stat_ones = stat_ones + 1;
    else
        answer = 0;
        stat_zeros = stat_zeros + 1;
    end
    data_vector = num2str(data_vector);
    symbols = length(data_vector) - l + 1;
    for i = 2:2:symbols
        data_vector(i) = '';
    end
    fprintf('ones = %d / zeros = %d\n', stat_ones, stat_zeros);
    fprintf(fid,'%s, %d\n', data_vector, answer);
end
fclose(fid);