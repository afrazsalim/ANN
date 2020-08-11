load threes -ascii
matrix = threes;
%colormap('gray');
%[rows,cols] = size(threes);
%k = 1;
%Compute the mean vector.
%mean_image = mean(threes);
%imagesc(reshape(mean_image,16,16),[0,1]);
%Compute the covariance matrix.
%cov_matrix = cov(threes);
data = threes;
data=data-repmat(mean(data,2),1,size(data,2));
% calculate eigenvectors (loadings) W, and eigenvalues of the covariance matrix
%step 3, covariance matrix
covariancematrix=cov(data);

%step 4, Finding Eigenvectors
[V,D] = eig(covariancematrix);
D=diag(D);
figure
plot(D);
hold on
title("Eigenvalues");
xlabel("Number of eigenvalues");
ylabel("Magnitude of the eigenvalues");
hold off
 for i = 4:-1:1
     disp(V(end-i,1:3));
 end

%maxeigval=V(:,find(D==max(D)));
maxeigval = V;
finaldata=(data * maxeigval');
finaldata = fliplr(finaldata);
org_data = (finaldata * inv(maxeigval)');
%org_data = org_data  + mean_image;
%imagesc(reshape(org_data(1,:),16,16),[0,1]);
%pca(data,'NumComponents',2);
% figure;
 %h    = [];
 %h(1) = subplot(2,2,1);
 %h(2) = subplot(2,2,2);
 %h(3) = subplot(2,2,3);
 %h(4) = subplot(2,2,4);
error_val = [];
for k = 1:256
    [coeff, score, latent, tsquared, explained, mu] = pca(threes,'NumComponents',k);
     reconstructed = score * coeff' + repmat(mu, 500, 1);
     %imagesc(reshape(reconstructed(1,:),16,16),[0,1]);
     sum((threes - reconstructed).^2);
     explained;
     approximationRank2 = score(:,1:k) * coeff(:,1:k)' + repmat(mu, 500, 1);
     %imagesc(reshape(approximationRank2(1,:),16,16),[0,1]);
     %subplot(2, 2, k);
    % hold on;
     %grid on
     %imagesc(reshape(approximationRank2(1,:),16,16),[0,1]);
    % image(reshape(approximationRank2(1,:),16,16),'Parent',h(k));
     %plot(approximationRank2(:, k), threes(:, k), 'x');
     error =  (sqrt(mean(mean((threes-reconstructed).^2))));
     error_val = [error_val,error];
     %title(sprintf('Components %d', k));
end
vec = D';
vec = fliplr(vec);
vec = cumsum(vec);
disp("Error " +error_val(1:end));
[rows,cols] = size(error_val);
figure
hold on
plot(1:cols,vec,'x-');
title("Comulative eigenvals");
xlabel("Eigenvalues");
ylabel("Commulative sum");
hold off
 
 