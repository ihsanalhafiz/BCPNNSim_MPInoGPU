/*

MIT License

Copyright (c) 2024 Anders Lansner, Naresh Ravichandran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <vector>
#include <string>
#include <random>
#include <iostream>

#include "Globals.h" 
#include "Pop.h"

using namespace std;
using namespace Globals;

int Pop::npop = 0;

Pop::Pop(int H, int M, std::string actfn, float again, int rank) {

    id = npop++;
    this->H = H;
    this->M = M;
    this->N = H * M;    
    this->actfn = actfn;
    this->again = again;
    this->rank = rank;
    eps = 1e-7;

    if (onthisrank()) {
        printf("\nNew Pop id = %-5d rank=%d mpi = %d/%d H = %-5d M = %-5d N = %-5d again=%-.3f", id, rank, Globals::mpi_world_rank, Globals::mpi_world_size, H, M, N, this->again);
        fflush(stdout);
    }
    
}

Pop::~Pop() {

    if (not onthisrank()) return;

    free(lgi);
    free(sup);
    free(act);
    
}

bool Pop::onthisrank() {

    if (rank<0 or rank>=Globals::mpi_world_size) { printf("Accessing rank = %d", rank); error("Pop::onthisrank", "Illegal rank!"); }

    return Globals::mpi_world_rank == rank;

}

typedef struct {
    unsigned int seed;
} cpuRandState;

void setup_noise_cpu(cpuRandState *state, int N) {
    // not applicable
}

void Pop::allocate_memory() {

    if (not onthisrank()) return;    

    lgi = (float*)malloc(N*sizeof(float));
    sup = (float*)malloc(N*sizeof(float));
    supinf = (float*)malloc(N*sizeof(float));
    act = (float*)malloc(N*sizeof(float));
    spkidx = (int*)malloc(H*sizeof(int));
    cumsum = (float*)malloc(N*sizeof(float));
    
    for (int i=0; i<N; i++) lgi[i] = 0; // logf(eps);
    for (int i=0; i<N; i++) sup[i] = logf(eps);
    for (int i=0; i<N; i++) supinf[i] = logf(eps);
    for (int i=0; i<N; i++) act[i] = eps;
    for (int h=0; h<H; h++) spkidx[h] = 0;
    for (int i=0; i<N; i++) cumsum[i] = 0;

    d_lgi = (float*)malloc(N*sizeof(float));
    d_sup = (float*)malloc(N*sizeof(float));
    d_supinf = (float*)malloc(N*sizeof(float));
    d_act = (float*)malloc(N*sizeof(float));
    d_spkidx = (int*)malloc(H*sizeof(int));
    d_cumsum = (float*)malloc(N*sizeof(float));

    memcpy(d_lgi, lgi, N*sizeof(float));
    memcpy(d_sup, sup, N*sizeof(float));
    memcpy(d_supinf, supinf, N*sizeof(float));
    memcpy(d_act, act, N*sizeof(float));
    memcpy(d_spkidx, spkidx, H*sizeof(int));
    memcpy(d_cumsum, cumsum, N*sizeof(float));

    // MPI stuff

    sendRequests = new MPI_Request[axons.size()];
    sendStats = new MPI_Status[axons.size()];
    recvRequests = new MPI_Request[dends.size()];
    recvStats = new MPI_Status[dends.size()];

    d_sendBuf = (float*)malloc(N*sizeof(float));
    for (int did=0; did<dends.size(); did++) {
        float *tmpptr;
        tmpptr = (float*)malloc(dends[did]->Ni*sizeof(float));
        d_recvBuf.push_back(tmpptr);
    }
}

void Pop::print_comm_summary() {

    if (not onthisrank()) return;

    double avgSendDur = 0, avgRecvDur = 0;
    
    if (sendNum!=0) avgSendDur = 1000000. * sendDur / sendNum;
    if (recvNum!=0) avgRecvDur = 1000000. * recvDur / recvNum;

    printf("\nPop rank = %d recvs num = %10d dur = %10.3f avgDur = %10.3fus sends num = %10d dur = %10.3fs avgDur = %10.3fus \n",
           rank, recvNum, recvDur, avgRecvDur, sendNum, sendDur, avgSendDur);
    fflush(stdout);

}

void Pop::store(std::string field, FILE* f) {

    if (not onthisrank()) return;

    if (field == "act") {        
        memcpy(act, d_act, N*sizeof(float));
        fwrite(act, sizeof(float), N, f);        
    } else if (field == "lgi") {
        memcpy(lgi, d_lgi, N*sizeof(float));
        fwrite(lgi, sizeof(float), N, f);
    } else if (field == "sup") {
        memcpy(sup, d_sup, N*sizeof(float));
        fwrite(sup, sizeof(float), N, f);
    } else if (field == "supinf") {
        memcpy(supinf, d_supinf, N*sizeof(float));
        fwrite(supinf, sizeof(float), N, f);   
    } else {
        printf("\nPop::store Invalid field!");
    }
            
}

float* Pop::getact() {

    if (not onthisrank()) return nullptr;
    
    return d_act;

}

void Pop::resetsup() {

    if (not onthisrank()) return;

    memset(d_supinf, 0, N*sizeof(float));
    memset(d_act, 0, N*sizeof(float));

}

void setinput_to_zero_kernel_cpu(float* lgi, float eps, int N) {
    for (int n = 0; n < N; n++) {
        lgi[n] = 0; // Or logf(eps) if needed
    }
}

void setinput_kernel_cpu(float* lgi, float* inp, float eps, int N) {
    for (int n = 0; n < N; n++) {
        lgi[n] = logf(inp[n] + eps);
    }
}

void Pop::setinput(float *d_inp = nullptr) {

    if (not onthisrank()) return;

    if (d_inp == nullptr) {
        setinput_to_zero_kernel_cpu(d_lgi, eps, N);
    } else {
        setinput_kernel_cpu(d_lgi, d_inp, eps, N);
    }
         
}

void Pop::sync_device() {

    if (not onthisrank()) return;

    //CHECK_HIP_ERROR(hipDeviceSynchronize());
    
}

void Pop::start_send() {

    if (not onthisrank()) return;

    if (axons.size()==0) return;    

    for (int aid=0; aid<axons.size(); aid++) {

        Prj *axon = axons[aid];

        if (this->rank == axon->pop_j->rank) {
            memcpy(axon->d_Xi, d_act, N*sizeof(float));
            sendRequests[aid] = MPI_REQUEST_NULL;
        } else {
            MPI_Isend(d_act, N, MPI_FLOAT, axon->pop_j->rank, 0, MPI_COMM_WORLD, &sendRequests[aid]);
        }

    }

}

void Pop::wait_and_end_send() {

    if (not onthisrank()) return;

    if (axons.size()==0) return;

    double t1 = MPI_Wtime();

    MPI_Waitall(axons.size(), sendRequests, sendStats); // wait till previous activity is fully sent
    
    double t2 = MPI_Wtime();

    sendDur += t2 - t1;
    sendNum += 1;

}

void Pop::start_recv() {

    if (not onthisrank()) return;

    if (dends.size()==0) return;

    for (int did=0; did<dends.size(); did++) {

        Prj *dend = dends[did];

        if (this->rank == dend->pop_i->rank) {
            recvRequests[did] = MPI_REQUEST_NULL;
        } else {
            MPI_Irecv(dend->d_Xi, dend->Ni, MPI_FLOAT, dend->pop_i->rank, 0, MPI_COMM_WORLD, &recvRequests[did]);
        }

    }

}

void Pop::wait_and_end_recv() {

    if (not onthisrank()) return;

    if (dends.size()==0) return;

    double t1 = MPI_Wtime(); 
 
    MPI_Waitall(dends.size(), recvRequests, recvStats); // wait till activity is fully received

    double t2 = MPI_Wtime();

    recvDur += t2 - t1;
    recvNum += 1;
    
}

void addtosupinf_kernel_cpu(float *supinf, float *inc, int N) {
    for (int n = 0; n < N; n++) {
        supinf[n] += inc[n];
    }
}

void calcsup_kernel_cpu(float *sup, float *supinf, int N) {
    for (int n = 0; n < N; n++) {
        sup[n] += (supinf[n] - sup[n]);
    }
}


void Pop::integrate() {

    /* integrate all dendrite prj's support into pop's support */

    if (not onthisrank()) return;

    addtosupinf_kernel_cpu(d_supinf, d_lgi, N);
    for (int did = 0; did < dends.size(); did++) {
        Prj *dend = dends[did];
        if (dend->bwgain < eps) continue;
        addtosupinf_kernel_cpu(d_supinf, dend->d_bwsup, N);
        // No need to check for errors as in CUDA version
    }
    calcsup_kernel_cpu(d_sup, d_supinf, N);
        
}

void inject_noise_kernel_cpu(float *sup, float nampl, int N) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Uniform distribution between 0 and 1
    unsigned seed = 1234;
    std::mt19937 rng(1234); // Standard mersenne_twister_engine seeded with rd()

    for (int n = 0; n < N; n++) {
        rng.discard(n); // Discard n random numbers
        sup[n] += nampl * dist(rng);
    }
}

void Pop::inject_noise(float nampl) {

    if (not onthisrank()) return;

    inject_noise_kernel_cpu(d_sup, nampl, N);
    
}

void softmax_kernel_cpu(float* sup, float* act, float again, int H, int M) {
    for (int h = 0; h < H; h++) {
        float maxsup = sup[M*h], sumexpsup = 0;  
        for (int m=0; m<M; m++) maxsup = max(maxsup, sup[M*h+m]); 
        for (int m=0; m<M; m++) act[M*h+m] = expf(sup[M*h+m] - maxsup);
        for (int m=0; m<M; m++) sumexpsup += act[M*h+m];  
        for (int m=0; m<M; m++) act[M*h+m] /= sumexpsup;
    }
}

void Pop::updact() {

    if (not onthisrank()) return;

    if (actfn == "softmax") {
        softmax_kernel_cpu(d_sup, d_act, again, H, M);
    } else {
        std::cerr << "Pop::updact - Unknown actfn" << std::endl;
    }
    
}

void Pop::resetncorrect() {

    if (not onthisrank()) return;
    
    ncorrect = 0;

}

void Pop::compute_accuracy(float *d_target) {

    if (not onthisrank()) return;

    float *d_pred = this->d_act;

    float *pred = new float[N];
    float *target = new float[N];
    
    memcpy(target, d_target, N*sizeof(float));
    memcpy(pred, d_pred, N*sizeof(float));
    
    ncorrect += argmax(target, 0, N) == argmax(pred, 0, N);
    printf("\n target = %d pred = %d ncorrect = %d", argmax(target, 0, N), argmax(pred, 0, N), ncorrect);

}

void Pop::settracc(int ntrpat) {

    if (not onthisrank()) return;

    tracc = float(ncorrect) / ntrpat * 100.;

}

void Pop::setteacc(int ntepat) {

    if (not onthisrank()) return;

    teacc = float(ncorrect) / ntepat * 100.;

}
