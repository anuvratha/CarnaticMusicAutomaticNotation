using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CarnaticMusicAutomaticNotation
{
    internal static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            var mutex = new Mutex(true, @"Local\CarnaticMusicAutomaticNotation.exe", out var mutexCreated);
            if (!mutexCreated)
            {
                mutex.Close();
                mutex.Dispose();
                MessageBox.Show("Another instance of this software is already running");
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
            mutex.Close();
            mutex.Dispose();
        }
    }
}
