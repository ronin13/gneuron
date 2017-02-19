using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.Research.DataParallelArrays;
using PA = Microsoft.Research.DataParallelArrays.ParallelArrays;
using DFPA = Microsoft.Research.DataParallelArrays.DisposableFloatParallelArray;
using FPA = Microsoft.Research.DataParallelArrays.FloatParallelArray;
using FP = Microsoft.Research.DataParallelArrays.Float4ParallelArray;
using DF = Microsoft.Research.DataParallelArrays.DisposableFloat4ParallelArray;

namespace Back
{
    class Network
    {
        private int ni;
        private int nh;
        private int no;
        private int numpat;

        private float[,] input;
        //private float[,] hidden;
        private float[,] output;

        private float[,] iwt;
        private float[,] owt;

        private float betah;
        private float betao;

        private float theta;
        private float tau;

        int traincycles;

        DFPA dinput,doutput;

        DFPA diwt, dowt;

        FPA dtheta, dtau;

        Random rand;

        public Network(int ini, int inh, int ino,int inumpat,int itraincycles)
        {
            ni = ini;
            no = ino;
            nh = inh;
            numpat = inumpat;
            traincycles = itraincycles;

            input = new float[numpat,ni];
            //hidden = new float[numpat, nh];
            output = new float[numpat, no];

            iwt = new float[ni, nh];
            owt = new float[nh, no];

            //theta = new float[numpat,nh];
            //tau = new float[numpat,no];
            dtheta = new FPA(theta, new int[] { numpat, nh });
            dtau = new FPA(tau, new int[] { numpat, no });

            rand = new Random();
            theta = tau = 0.1f;
            betah = betao = 0.02f; 

        }

        public void start()
        {
            init();

            PA.InitGPU();

            

            diwt = new DFPA(iwt);
            dowt = new DFPA(owt);

            while (traincycles-- > 0)
            {
                dinput = new DFPA(input);
                doutput = new DFPA(output);
                run();
            }

            cleanup();
            
            PA.UnInit();
          }
        private void run()
        {
           

            FPA t1 = PA.Add(PA.InnerProduct(dinput, diwt),theta);  // summation and theta 

            FPA ohidden = PA.Reciprocal(PA.Add(PA.Pow2(PA.Negate(t1)), 1.0f)); //applying sigmoid function 

            FPA t2 = PA.Add(PA.InnerProduct(ohidden, dowt), tau); // summation and tau

            FPA ooutput = PA.Reciprocal(PA.Add(PA.Pow2(PA.Negate(t2)), 1.0f));

            FPA oerror = PA.Subtract(doutput, ooutput); //numpat no 

            FPA herror = PA.Transpose(PA.InnerProduct(dowt, PA.Transpose(oerror, new int[] { 1, 0 })), new int[] { 1, 0 }); // doubtful transpose
            
            herror = PA.Multiply(PA.Multiply(PA.Subtract(1.0f, t1),t1), herror);

            FPA _owt = PA.Add(dowt, PA.Multiply(PA.InnerProduct(PA.Transpose(t1,new int[]{1,0}),oerror), betao)); // orig no transpose


            FPA _iwt = PA.Multiply(PA.InnerProduct(PA.Transpose(dinput,new int[]{1,0}),herror), betah) ; //original dinput herror and no transpose

            dtau = PA.Add(PA.Multiply(betao, oerror),dtau);

            dtheta = PA.Add(PA.Multiply(betah, herror),dtheta); //orig oerror

            PA.ToArray(_owt, out owt);
            PA.ToArray(_iwt, out iwt);

            cleanup();
            diwt = new DFPA(iwt);
            dowt = new DFPA(owt);


        }
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
        private void cleanup()
        {
            dinput.Dispose();
            doutput.Dispose();
            diwt.Dispose();
            dowt.Dispose();

        }

    }
}
