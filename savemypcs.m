 pathfigs='/Users/wenlin_wu/Desktop/'
 hf=figure('Name','genotype 3 vs 4 ');
 pca1=2
 pca2=3
 gen1='0'
 %gen2='4'
 my1=test2(:,pca1)
 my2=test2(:,pca2)
 scatter(my1,my2,[],test2(:,4),'*');
 %title(['genotype 3 vs 4 onlye3e4' num2str(gen1) ' PC' num2str(pca1) ' &' ' PC' num2str(pca2)])
 title(['genotype 3 vs 4 ' ' PC' num2str(pca1) ' &' ' PC' num2str(pca2)])

 
 
 %legend({num2str(gen1)} , {num2str(gen2)} )


 colormap(cmap)
 myxlabel=['pc' num2str(pca1)]
 myylabel=['pc' num2str(pca2)]
 
xlabel(myxlabel, 'FontSize', 16, 'FontWeight', 'bold');
ylabel(myylabel ,'FontSize', 16, 'FontWeight', 'bold');

%colorbar;

set(gca,'FontSize',16,'FontName','FixedWidth', 'FontWeight', 'bold');


%fname=char([pathfigs  'sex in Genotype' num2str(gen1) '_PC' num2str(pca1) '_' num2str(pca2) '.png'])
fname=char([pathfigs  'genotype 3 vs 4 ' '_PC' num2str(pca1) '_' num2str(pca2) '.png'])

saveas(hf, fname,'png');

export_fig(fname, '-png', '-r600');