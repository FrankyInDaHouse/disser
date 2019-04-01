if exist('csv_table.csv', 'file')
    clear;
    fclose('all');
    delete('csv_table.csv');
end
textHeader = "received_seq, is_cw";
fid = fopen('csv_table.csv','w');
fprintf(fid,'%s\n',textHeader);

% Parameters of code
length = 3;
amount_of_seq = 2^length;
cw_cnt = 0;
ncw_cnt = 0;

for iterator = 1 : amount_of_seq
    value = iterator - 1;
    data_vector = fliplr( de2bi(value, length) );

    if (mod(sum(data_vector),2) == 0)
        is_cw = 1;
        cw_cnt = cw_cnt + 1;
    else
        is_cw = 0;
        ncw_cnt = ncw_cnt + 1;
    end
    
    data_vector = num2str(data_vector);
    fprintf(fid,'%s, %d\n', data_vector, is_cw);
end
fclose(fid);