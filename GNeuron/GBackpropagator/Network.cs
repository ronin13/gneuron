#region Disclaimer
/** By: Raghavendra D Prabhu 
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

    This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
#endregion
#region Imports
using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;


/* The imports required for GPU operations */
using Microsoft.Research.DataParallelArrays;
using PA = Microsoft.Research.DataParallelArrays.ParallelArrays;
using DFPA = Microsoft.Research.DataParallelArrays.DisposableFloatParallelArray;
using FPA = Microsoft.Research.DataParallelArrays.FloatParallelArray;
using FP = Microsoft.Research.DataParallelArrays.Float4ParallelArray;
using DF = Microsoft.Research.DataParallelArrays.DisposableFloat4ParallelArray;
#endregion

namespace GBackpropagator
{
    public class Network
    {
        #region Declarations
        #region classdecl

        /* FPA - Floating Parallel Array 
         * DFPA - Disposable Floating Parallel Array 
         * PA - Parallel Arrays : Class which defines static methods for all the operations 
         */

        /* Size of Input layer */
        private int ni; 

        /* Size of Hidden layer */
        private int nh;

        /* Size of Output layer */
        private int no;

        /*Number of patterns */
        private int numpat;

        /* Input pattern
         * It is of dimension number of patterns X number of inputs 
         */

        private float[,] input;

        /* Expected output 
         * It is of dimension  number of patterns X number of outputs
         */
        private float[,] output;

        /*Weights between input and hidden layer 
         * It is of dimension 
         *  size of input X size of hidden 
         */

        private float[,] iwt;

        /*Weights between hidden and output layer 
         *  It is of dimension 
         *  size of hidden X size of output
         */
        private float[,] owt;

        /* Learning parameter of hidden layer */
        private float betah;

        /*Learning parameter for output layer */
        private float betao;

        /*Threshold for hidden layer */
        private float theta;

        /*Threshold for output layer*/
        private float tau;

        /*Number of training cycles */
        private int traincycles;

        private int numcycles;

        /* All functions hereforth prefixed with 'd' represent Disposable FPA or just FPA of their counterparts */
        DFPA dinput,doutput;

        DFPA diwt, dowt;

        FPA dtheta, dtau,derror;

        Random rand;

        private long timend = 0;
        private long timbeg = 0;

        private double _timtaken;
        #endregion

        #region WinAPI
        /*
         * Functions to make accurate time measurements          
         */
        [DllImport("kernel32.dll")]
        extern static short QueryPerformanceCounter(ref long x);
        [DllImport("kernel32.dll")]
        extern static short QueryPerformanceFrequency(ref long x);
        #endregion
        #endregion

        #region Accessors
        public string timetaken
        {
            get{ return _timtaken.ToString();}
        }
        public int inlen
        {
            get { return ni; }
        }
        public int outlen
        {
            get { return no; }
        }
        public string cycles
        {
            get { return numcycles.ToString(); }
        }

        #endregion

        #region Constructor
        public Network(int ini, int inh, int ino,int inumpat,int itraincycles)
        {
            ni = ini;
            no = ino;
            nh = inh;
            numpat = inumpat;
            traincycles = itraincycles;

            input = new float[numpat,ni];
            output = new float[numpat, no];

            iwt = new float[ni, nh];
            owt = new float[nh, no];

            dtheta = new FPA(theta, new int[] { numpat, nh });
            dtau = new FPA(tau, new int[] { numpat, no });

            rand = new Random();
            theta = tau = 0.35f;
            betao = 0.2f;
            betah = 0.15f;

        }
        #endregion
        #region Main Part
        /*
         * Entry Function 
         */
        public void start()
        {
            /* Initialisation of all layers*/
            init();

            /*Normalisation of weights */
            normali();
            normalo();

            /*Initialisation of GPU*/
            PA.InitGPU();
            
            /*Measurement starts*/
            QueryPerformanceCounter(ref timbeg);

            diwt = new DFPA(iwt);
            dowt = new DFPA(owt);

            dinput = new DFPA(input);
            doutput = new DFPA(output);

            /* Minimum permissible error */
            derror = PA.Abs(PA.Multiply(doutput, 0.01f));

            while (traincycles > 0)
            {
               traincycles--;
               numcycles++;
               run();
            }
            
            long freq = 0;
            /*Measurement ends */
            QueryPerformanceCounter(ref timend);
            QueryPerformanceFrequency(ref freq);
            _timtaken = (timend - timbeg) * 1.0 / freq;
                     
          }

        /*
         *Function which performs all the GPU operations  
         */ 
        private void run()
        {
            /* Note : Inner product --- Matrix multiplication 
             *        Multiply -- Element by element multiplication */     

            FPA t1 = PA.Add(PA.InnerProduct(dinput, diwt),dtheta);  
         
            /* ohidden is the output of hidden layer
            Only Sigmoid function is used for timebeing */
            FPA ohidden = PA.Reciprocal(PA.Add(PA.Pow(new FPA(2.71828f,new int[]{numpat,nh}),PA.Negate(t1)), 1.0f)); 
         
            FPA t2 = PA.Add(PA.InnerProduct(ohidden, dowt), dtau);

            /* ooutput is the "actual" output of hidden layer
               Only Sigmoid function is used for timebeing */
            FPA ooutput = PA.Reciprocal(PA.Add(PA.Pow(new FPA(2.71828f,new int[]{numpat,no}),PA.Negate(t2)), 1.0f));
            
            /* Error between expected and actual */
            FPA oerror = PA.Subtract(doutput, ooutput); 

            /* Checking if error has fallen below 1% if so terminatinf further cycles */
            BoolParallelArray b = PA.All(PA.CompareGreater(derror,PA.Abs(oerror)),1);
            b = PA.All(b);
            bool[] bt;
            PA.ToArray(b, out bt);
            if (bt[0] == true)
                traincycles = 0;
                    
            /* herror is the error in the hidden layer */
            FPA herror = PA.Transpose(PA.InnerProduct(dowt, PA.Transpose(oerror, new int[] { 1, 0 })), new int[] { 1, 0 }); 
            
            herror = PA.Multiply(PA.Multiply(PA.Subtract(1.0f, ohidden),ohidden), herror);

            /* Weights between hidden  and output layer being updated */  
            FPA _owt = PA.Add(PA.Multiply(PA.InnerProduct(PA.Transpose(ohidden,new int[]{1,0}),oerror),betao),dowt);

            /* Weights between input  and hidden layer being updated */  
            FPA _iwt = PA.Add(PA.Multiply(PA.InnerProduct(PA.Transpose(dinput,new int[]{1,0}),herror), betah),diwt) ; 

            /*Updating threshold for output layer */
            dtau = PA.Add(PA.Multiply(betao, oerror),dtau);

            /*Updating threshold for hidden layer */
            dtheta = PA.Add(PA.Multiply(betah, herror),dtheta);

            /* Casting the Parallel arrays to normal arrays */
            PA.ToArray(_owt, out owt);
            PA.ToArray(_iwt, out iwt);

            /* Rebuilding the disposable arrays from newly formed arrays */
            diwt = new DFPA(iwt);
            dowt = new DFPA(owt);


        }
        #endregion

        #region Network Testing
        public float[] Test(float[] iinput)
        {        

            float[,] tinput = new float[1, ni];
            for (int i = 0; i < ni; i++)
                tinput[0, i] = iinput[i];

            dinput = new DFPA(tinput);
            diwt = new DFPA(iwt);
            dowt = new DFPA(owt);

            dtheta = PA.Section(dtheta, new Slice(0, 1), new Slice(0,nh));
            dtau = PA.Section(dtau, new Slice(0, 1), new Slice(0,no)); 

            FPA t1 = PA.Add(PA.InnerProduct(dinput, diwt), dtheta);  
            FPA ohidden = PA.Reciprocal(PA.Add(PA.Pow2(PA.Negate(t1)), 1.0f)); 
            FPA t2 = PA.Add(PA.InnerProduct(ohidden, dowt), dtau); 

            FPA ooutput = PA.Reciprocal(PA.Add(PA.Pow2(PA.Negate(t2)), 1.0f));

            float[,] output;
            float[] routput = new float[no];
            PA.ToArray(ooutput, out output);

            for (int i = 0; i < no; i++)
                routput[i] = output[0, i];
                      
            /*Disposable Floating arrays need to be explicitly "disposed" */
            dinput.Dispose();
            diwt.Dispose();
            dowt.Dispose();
            doutput.Dispose();

            /*Releasing all GPU Resources*/
            PA.UnInit();

            return routput;

        }
        public void Terminate()
        {
            dinput.Dispose();
            diwt.Dispose();
            dowt.Dispose();
            doutput.Dispose();
            PA.UnInit();
        }

        #endregion

        #region Misc
        private void init()
        {
            for (int i = 0; i < numpat; i++)
            {
                for (int j = 0; j < ni; j++)
                {
                    input[i, j] = (float)rand.NextDouble();
                }
            }
            for (int i = 0; i < numpat; i++)
            {
                for (int j = 0; j < no; j++)
                {
                    output[i, j] = (float)rand.NextDouble();
                }
            }

            for (int i = 0; i < ni; i++)
            {
                for (int j = 0; j < nh; j++)
                {
                    iwt[i,j] = (float)rand.NextDouble();
                    
                  
                }
            }
            for (int i = 0; i < nh; i++)
            {
                for (int j = 0; j < no; j++)
                {
                    owt[i, j] = (float)rand.NextDouble();
                
                }
            } 
        }

        /*
         * Cleaning up Disposable arrays 
         */
        private void cleanup()
        {
            dinput.Dispose();
            doutput.Dispose();
            diwt.Dispose();
            dowt.Dispose();

        }

        #region Normalisation
        public void normali()
        {
            int j = 0;
            int i = 0;
            float temp;
            float acc;


            temp = 0.0f;
            acc = 0.0f;

            while (i < ni)
            {
                j = 0;
                while (j < nh)
                {
                    acc = iwt[i, j] * iwt[i, j];
                    temp += acc;
                    j = j + 1;
                }
                temp = (float)Math.Sqrt(temp);
                temp = (float)(1.0f / temp);
                j = 0;

                while (j < nh)
                {
                    iwt[i, j] = iwt[i, j] * temp;
                    j = j + 1;
                }
                i++;
            }


        }
        public void normalo()
        {
            int i = 0;
            int j = 0;
            float temp;
            float acc;


            temp = 0.0f;
            acc = 0.0f;

            while (i < nh)
            {
                j = 0;
                while (j < no)
                {
                    acc = owt[i, j] * owt[i, j];
                    temp += acc;
                    j = j + 1;
                }
                temp = (float)Math.Sqrt(temp);
                temp = (float)(1.0f / temp);
                j = 0;

                while (j < no)
                {
                    owt[i, j] = owt[i, j] * temp;
                    j = j + 1;
                }
                i++;
            }
        }
        #endregion
        #endregion
    }
}
