function sendTFMail(TBaseP,ParamsS,Txt)
try
    CurTrainDirbase=[ ParamsS.SessionNameBase '_' ParamsS.dataset '__'];
    LastDir=getMostRecentDirWithBase(TBaseP,CurTrainDirbase);
    D2=dir([TBaseP LastDir '/*.png']);
    D2N={D2.name}';
    for i=1:numel(D2)
        D2Idx(i)=str2num(D2N{i}(6:end-8));
    end
    [~,MI2]=max(D2Idx);
    LastImgFN=[TBaseP LastDir filesep D2N{MI2}];
    
    gSendMail(CurTrainDirbase,Txt,{LastImgFN});
catch
end