if exist('csv_table.csv', 'file')
    clear;
    fclose('all');
    delete('csv_table.csv');
end
textHeader = "received_seq, is_cw";
fid = fopen('csv_table.csv','w');
fprintf(fid,'%s\n',textHeader);

% Parameters of code
%code_params = [7 4]; % Hamming code (7,4)
code_params = [15 11]; % Hamming code (15,11)

n = code_params(1);
k = code_params(2);
r = n - k;

if ((n == 7) && (k == 4)) % frow wiki
    G = [1 0 0 0 1 0 1 ;
         0 1 0 0 1 1 1 ;
         0 0 1 0 1 1 0 ;
         0 0 0 1 0 1 1 ;];
end
if ((n == 3) && (k == 1))
        G = [1 1 1];
end
if ((n == 15) && (k == 11))
        G = [ 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0;
              0 1 0 0 0 0 0 0 0 0 0 1 0 1 0;
              0 0 1 0 0 0 0 0 0 0 0 0 1 1 0;
              0 0 0 1 0 0 0 0 0 0 0 1 1 1 0;
              0 0 0 0 1 0 0 0 0 0 0 1 0 0 1;
              0 0 0 0 0 1 0 0 0 0 0 0 1 0 1;
              0 0 0 0 0 0 1 0 0 0 0 1 1 0 1;
              0 0 0 0 0 0 0 1 0 0 0 0 0 1 1;
              0 0 0 0 0 0 0 0 1 0 0 1 0 1 1;
              0 0 0 0 0 0 0 0 0 1 0 0 1 1 1;
              0 0 0 0 0 0 0 0 0 0 1 1 1 1 1;];
end

amount_of_codewords = 2^k;
amount_of_errors = 2^r;

disp('Fill codewords and errors')
%%%%%%%%%%%%%%%%%FILL CODEWORDS AND ERRORS
codewords{amount_of_codewords} = zeros();
for iterator = 1 : amount_of_codewords
    value = iterator - 1;
    data_vector = fliplr( de2bi(value, k) );
    codewords{iterator} = mod(data_vector * G, 2);
end
errors{amount_of_errors} = zeros();
errors{1} = zeros(1, n);
for iterator = 0 : n - 1
    value = 2 ^ iterator;
    errors{iterator + 2} = fliplr( de2bi(value, n) );
end
%%%%%%%%%%%%%%%%%FILL CODEWORDS AND ERRORS

disp('Standart table creation')
%%%%%%%%%%%%%%%%%STANDART TABLE CREATION
standart_table{amount_of_codewords, amount_of_errors} = zeros();
for curr_cw = 1 : amount_of_codewords
    for curr_error = 1 : amount_of_errors
        standart_table{curr_cw, curr_error} = mod(codewords{curr_cw} + errors{curr_error} , 2);
    end
end
%%%%%%%%%%%%%%%%%STANDART TABLE CREATION

disp('CSV table creation')
%%%%%%%%%%%%%%%%%CSV TABLE CREATION
csv_table{amount_of_codewords * 2 * n, 2} = zeros();
write_to_row = 1;
for standart_table_row = 1 : size(standart_table,1)
    for standart_table_col = 2 : size(standart_table,2)
        % cw
        csv_table{write_to_row, 1} = standart_table{standart_table_row, 1};
        csv_table{write_to_row, 2} = 1;
        
        csv_table{write_to_row, 1} = num2str(cell2mat(csv_table(write_to_row, 1)));
        % making only one space between symbols
        symbols = length(csv_table{write_to_row, 1}) - n + 1;
        for i = 2:2:symbols
            csv_table{write_to_row, 1}(i) = '';
        end
        fprintf(fid,'%s, %d\n', csv_table{write_to_row, 1}, csv_table{write_to_row, 2});
        %%%%%%%%%%%%%%%%%%%
        write_to_row = write_to_row + 1;
        %%%%%%%%%%%%%%%%%%%
        % cw + err
        csv_table{write_to_row, 1} = standart_table{standart_table_row, standart_table_col};
        csv_table{write_to_row, 2} = 0;
        
        csv_table{write_to_row, 1} = num2str(cell2mat(csv_table(write_to_row, 1)));
        % making only one space between symbols
        symbols = length(csv_table{write_to_row, 1}) - n + 1;
        for i = 2:2:symbols
            csv_table{write_to_row, 1}(i) = '';
        end
        fprintf(fid,'%s, %d\n', csv_table{write_to_row, 1}, csv_table{write_to_row, 2});
        %%%%%%%%%%%%%%%%%%%
        write_to_row = write_to_row + 1;
        %%%%%%%%%%%%%%%%%%%
        fprintf('writing row = %d\n', write_to_row);
    end
end
fid = fclose(fid);
%%%%%%%%%%%%%%%%%CSV TABLE CREATION




