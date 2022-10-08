#include "common.h"


FileGetter::FileGetter(char* folderin,char* ext){		
	strcpy(folder,folderin);
	char folderstar[MAX_PATH];
    if( !ext ) strcpy(ext,"*");
	sprintf(folderstar,"%s\\*.%s",folder,ext);

	//skip .
	//FindNextFileA(hfind,&found);		
}

int FileGetter::getNextFile(char* fname){
//	if (!hasFiles)
//		return 0;
//	//skips .. when called for the first time
//	if( first )
//	{
//		strcpy(fname, found.cFileName);
//		first = 0;
//		return 1;
//	}
//	else{
//		chk=FindNextFileA(hfind,&found);
//		if (chk)
//			strcpy(fname, found.cFileName);
//		return chk;
//	}
}

int FileGetter::getNextAbsFile(char* fname){
//	if (!hasFiles)
//		return 0;
//	//skips .. when called for the first time
//	if( first )
//	{
//		sprintf(fname, "%s\\%s", folder, found.cFileName);
//		first = 0;
//		return 1;
//	}
//	else{
//		chk=FindNextFileA(hfind,&found);
//		if (chk)
//			sprintf(fname, "%s\\%s", folder, found.cFileName);
//		return chk;
//	}
}

char* FileGetter::getFoundFileName(){
//	if (!hasFiles)
//		return 0;
//	return found.cFileName;
}


int openFileDlg(char* fname)
{

    FILE *f = popen("zenity --file-selection", "r");
    fgets(fname, MAX_PATH, f);
    fname[strlen(fname) - 1] = 0;
    printf("Chosen path: [%s]\n", fname);
    return strcmp(fname, "");

}

int openFolderDlg(char *folderName)
{
    FILE *f = popen("zenity --file-selection", "r");
    fgets(folderName, MAX_PATH, f);
    folderName[strlen(folderName) - 1] = 0;
    printf("Chosen path: [%s]\n", folderName);
    return strcmp(folderName, "");
}

void resizeImg(Mat src, Mat &dst, int maxSize, bool interpolate)
{
	double ratio = 1;
	double w = src.cols;
	double h = src.rows;
	if (w>h)
		ratio = w/(double)maxSize;
	else
		ratio = h/(double)maxSize;
	int nw = (int) (w / ratio);
	int nh = (int) (h / ratio);
	Size sz(nw,nh);
	if (interpolate)
		resize(src,dst,sz);
	else
		resize(src,dst,sz,0,0,INTER_NEAREST);
}