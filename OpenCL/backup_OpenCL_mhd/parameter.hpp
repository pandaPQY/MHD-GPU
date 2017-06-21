#ifndef PARAMETER_H_ 
#define PARAMETER_H_ 

#define n 128//32//64//64//16//32//32//64//128//64//32//32//16//128//32//16
#define box_nx (n)
#define box_ny (n)
#define box_nz (n)
#define SIZEofWORKGROUP (n)
#define nu (5)
#define nb (3)
#define total_cell_number (box_nx*box_ny*box_nz)

#define TimeTheTime 1
#define OutputEveryTimestep 1

#define TimeTheTime_all 1
#define TimeTheTime_fluid 0
#define TimeTheTime_magnetic 0
#define TimeTheTime_transpose 0
#define TimeTheTime_cfl 0

#endif	// PARAMETER_H_
