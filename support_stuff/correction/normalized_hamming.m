if exist('csv_table.csv', 'file')
    clear;
    fclose('all');
    delete('csv_table.csv');
end
textHeader = "received_seq, corr_result";
fid = fopen('csv_table.csv','w');
fprintf(fid,'%s\n',textHeader);

% Parameters of code
% code_params = [7 4]; % Hamming code (7,4)
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
% 1st column (all possible variants)
csv_table{2^n, 2} = zeros();
csv_table(:, 1) = standart_table(:);
% 2nd column (dec representation)
for row = 1 : 2^n
    fprintf('writing row = %d\n', row);
    
    [codeword_row, error_col] = find( cellfun( @(x) isequal(x, csv_table{row, 1}), standart_table ) );
    
    csv_table{row, 2} = cell2mat(        standart_table(codeword_row, 1)              );
    csv_table{row, 2} = csv_table{row, 2}(1:k);
    
    %conversion to strings
    csv_table{row, 1} = num2str(cell2mat(csv_table(row, 1)));
    % making only one space between symbols
    symbols = length(csv_table{row, 1}) - n + 1;
    for i = 2:2:symbols
        csv_table{row, 1}(i) = '';
    end
    
    csv_table{row, 2} = num2str(cell2mat(csv_table(row, 2)));
    % making only one space between symbols
    symbols = length(csv_table{row, 2}) - k + 1;
    for i = 2:2:symbols
        csv_table{row, 2}(i) = '';
    end
    
    fprintf(fid,'%s, %s\n', csv_table{row, 1}, csv_table{row, 2});
end
fid = fclose(fid);
%%%%%%%%%%%%%%%%%CSV TABLE CREATION