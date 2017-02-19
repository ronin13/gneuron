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

using GBackpropagator;
#endregion
namespace Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 5)
            {
                Console.WriteLine("Usage: Sample.exe inputs hidden outputs numpatterns numiterations");
                return;
            }
            Network n = new Network(Int32.Parse(args[0]), Int32.Parse(args[1]), Int32.Parse(args[2]), Int32.Parse(args[3]), Int32.Parse(args[4]));
            n.start();

            Console.WriteLine("*******************************");
            Console.WriteLine("Training done and is ready for testing");
            Console.WriteLine();
            Console.WriteLine("Number of cycles -------->" + n.cycles);
            Console.WriteLine();
            Console.WriteLine("Total time taken -----> " +n.timetaken);
            Console.WriteLine("****************************");

            Console.WriteLine("Enter sample input (Press 1 for random generation 2 for user input 0 to quit) ");

            #region Network testing
            int ni = n.inlen;
            int no = n.outlen;
            float[] input = new float[ni];
            Random r = new Random();
           

            if (Int32.Parse(Console.ReadLine()) == 1)
            {
                for (int i = 0; i < ni; i++)
                    input[i] = (float)r.NextDouble();
            }
            else if (Int32.Parse(Console.ReadLine()) == 2)
            {
                for (int i = 0; i < ni; i++)
                    input[i] = float.Parse(Console.ReadLine());
            }
            else
            {
                n.Terminate();
                return;
            }
            float[] output = n.Test(input);

            Console.WriteLine("The output is ");
            for (int i = 0; i < no; i++)
                Console.Write(output[i].ToString()+" ");
            Console.WriteLine();
            #endregion

        }
    }
}
