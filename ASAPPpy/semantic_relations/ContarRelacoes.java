package pt.uc.dei.sim.assin;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import pt.uc.dei.assin.xml.Par;
import pt.uc.dei.ontopt.estruturas.GrafoND;
import pt.uc.dei.ontopt.estruturas.Triplo;
import pt.uc.dei.ontopt.estruturas.TriplosIndexadosTipo;
import pt.uc.dei.ontopt.processadores.ProcessadorTriplosTermos;
import pt.uc.dei.sim.frases.UtilidadesSimilaridade;

public class ContarRelacoes extends UtilidadesSimilaridade
{
	//protected static final String DIR_OUT = "features_csv_201711";
	protected static final String DIR_OUT = null;

	protected static final String[] FILTRO_ANTONIMIA = {
			"ANTONIMO_N_DE", "ANTONIMO_V_DE", "ANTONIMO_ADJ_DE", "ANTONIMO_ADV_DE"
	};
	protected static List<String> filtroAntonimia = Arrays.asList(FILTRO_ANTONIMIA);

	protected static final int SINONIMIA = 0;
	protected static final int HIPERONIMIA = 1;
	protected static final int ANTONIMIA = 2;
	protected static final int OUTRAS = 3;

	private static boolean NORMALIZAR_COM_TAMANHO_FRASES = true; 

	private GrafoND<String> sinonimia;
	private GrafoND<String> antonimia;
	private GrafoND<String> hiperonimia;
	private GrafoND<String> outrasRels;

	public ContarRelacoes(TriplosIndexadosTipo<String> triplos)
	{
		Set<Triplo<String>> triplosSinonimia = new HashSet<>();
		if(triplos.temTriplosDoTipo("SINONIMO_N_DE"))
			triplosSinonimia.addAll(triplos.getTriplosDoTipo("SINONIMO_N_DE"));
		if(triplos.temTriplosDoTipo("SINONIMO_V_DE"))
			triplosSinonimia.addAll(triplos.getTriplosDoTipo("SINONIMO_V_DE"));
		if(triplos.temTriplosDoTipo("SINONIMO_ADJ_DE"))
			triplosSinonimia.addAll(triplos.getTriplosDoTipo("SINONIMO_ADJ_DE"));
		if(triplos.temTriplosDoTipo("SINONIMO_ADV_DE"))
			triplosSinonimia.addAll(triplos.getTriplosDoTipo("SINONIMO_ADV_DE"));
		sinonimia = grafoComTriplos(triplosSinonimia);

		Set<Triplo<String>> triplosAntonimia = new HashSet<>();

		if(triplos.temTriplosDoTipo("ANTONIMO_N_DE"))
			triplosAntonimia.addAll(triplos.getTriplosDoTipo("ANTONIMO_N_DE"));
		if(triplos.temTriplosDoTipo("ANTONIMO_V_DE"))	
			triplosAntonimia.addAll(triplos.getTriplosDoTipo("ANTONIMO_V_DE"));
		if(triplos.temTriplosDoTipo("ANTONIMO_ADJ_DE"))
			triplosAntonimia.addAll(triplos.getTriplosDoTipo("ANTONIMO_ADJ_DE"));
		if(triplos.temTriplosDoTipo("ANTONIMO_ADV_DE"))
			triplosAntonimia.addAll(triplos.getTriplosDoTipo("ANTONIMO_ADV_DE"));
		antonimia = grafoComTriplos(triplosAntonimia);

		Set<Triplo<String>> triplosHiperonimia = new HashSet<>();
		if(triplos.temTriplosDoTipo("HIPERONIMO_DE"))
			triplosHiperonimia.addAll(triplos.getTriplosDoTipo("HIPERONIMO_DE"));
		if(triplos.temTriplosDoTipo("HIPERONIMO_ACCAO_DE"))
			triplosHiperonimia.addAll(triplos.getTriplosDoTipo("HIPERONIMO_ACCAO_DE"));
		hiperonimia = grafoComTriplos(triplosHiperonimia);

		Set<String> filtroExclusivo = new HashSet<>();
		filtroExclusivo.addAll(filtroSinHiper);
		filtroExclusivo.addAll(filtroAntonimia);
		outrasRels = grafoComTriplos(triplos.getTriplosTodos(), filtroSinHiper);
	}

	private GrafoND<String> grafoComTriplos(Set<Triplo<String>> triplos)
	{
		GrafoND<String> grafo = new GrafoND<String>();
		for(Triplo<String> t : triplos)
			grafo.addArco(t.getArg1(), t.getArg2());

		return grafo;
	}

	private GrafoND<String> grafoComTriplos(Set<Triplo<String>> triplos, Collection<String> filtroExclusivo)
	{
		GrafoND<String> grafo = new GrafoND<String>();
		for(Triplo<String> t : triplos)
			if(!filtroExclusivo.contains(t.getPredicado()))
				grafo.addArco(t.getArg1(), t.getArg2());

		return grafo;
	}

	public double[] contagem(Par par)
	{
		double[] linha = new double[4];

		List<String>[] frasesAnotadas = (List<String>[])preparaParRelax(par, true, true);

/*		System.out.println("\tIntersec: "
				+SetUtils.getIntersection(new HashSet<>(frasesAnotadas[0]), new HashSet<>(frasesAnotadas[1])));
		System.out.println("\tReun: "
				+SetUtils.getUnion(new HashSet<>(frasesAnotadas[0]), new HashSet<>(frasesAnotadas[1])));
		System.out.println("\tJaccard: "
				+SetUtils.getJaccardCoefficient(new HashSet<>(frasesAnotadas[0]), new HashSet<>(frasesAnotadas[1])));*/

		for(String p1 : frasesAnotadas[0])
		{
			for(String p2 : frasesAnotadas[1])
			{
				if(sinonimia.contemArco(p1, p2))
				{
					System.out.println(p1+" sinonimo-de "+p2);
					linha[SINONIMIA] += 1;
				}
				if(hiperonimia.contemArco(p1, p2))
				{
					System.out.println(p1+" hiperonimo-de "+p2);
					linha[HIPERONIMIA] += 1;
				}
				if(antonimia.contemArco(p1, p2))
				{
					System.out.println(p1+" antonimo-de "+p2);
					linha[ANTONIMIA] += 1;
				}
				if(outrasRels.contemArco(p1, p2))
				{
					System.out.println(p1+" relacionado-com "+p2);
					linha[OUTRAS] += 1;
				}
			}
		}

		if(NORMALIZAR_COM_TAMANHO_FRASES)
		{
			normaliza(linha, frasesAnotadas[0].size()+frasesAnotadas[1].size());
		}

		return linha;
	}

	private void normaliza(double[] original, int factor)
	{
		for(int i = 0; i < original.length; i++)
			original[i] = original[i] / (double)factor;
	}

	public static void main(String args[])
	{
		ParserParesASSIN parser = new ParserParesASSIN();
		List<Par> pares = null;
		try {
			pares = parser.processaPares(args[0]);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(pares.size()+" pares!");


		ProcessadorTriplosTermos ptt = new ProcessadorTriplosTermos();

		for(int i = 1 ; i < args.length ; i++)
		{
			File dir = new File(args[i]);

			if(dir.isDirectory())
			{
				for(String fich : dir.list())
				{
					String fName = dir + File.separator + fich;
					if(!fName.endsWith(".txt"))
						continue;

					fazContagem(args[0], pares, fName, ptt);
				}
			}
			else
			{
				fazContagem(args[0], pares, args[i], ptt);
			}
		}
	}

	private static void fazContagem(String col, List<Par> pares, String fTriplos, ProcessadorTriplosTermos ptt)
	{
		PrintStream out = null;

		System.out.println("\t"+fTriplos);
		TriplosIndexadosTipo<String> triplos =
				ptt.getTriplosIndexadosPorTipo(fTriplos, false, false);
		ContarRelacoes contador = new ContarRelacoes(triplos);

		try {
			out = DIR_OUT == null ? System.out : new PrintStream(DIR_OUT+"/"+idColecao(col)+(NORMALIZAR_COM_TAMANHO_FRASES ? "_contagem_norm_" : "_contagem_")+idRede(fTriplos)+".csv");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		for(int i = 0; i < pares.size(); i++)
		{
			System.out.println("##### Par "+(i+1)+" #####");
			double[] linha = contador.contagem(pares.get(i));
			out.print(pares.get(i).getId()+"\t"+linha[SINONIMIA]+"\t"+linha[HIPERONIMIA]+"\t"+linha[ANTONIMIA]+"\t"+linha[OUTRAS]);
			out.println();
		}

		out.close();
	}

	/*	public static void main(String args[])
	{
		ProcessadorTriplosTermos ptt = new ProcessadorTriplosTermos();
        TriplosIndexadosTipo<String> triplos = ptt.getTriplosIndexadosPorTipo(args[0], false, false);
        ContarRelacoes contador = new ContarRelacoes(triplos);

        //String h = "A pegada ecológica é de 4,5 hectares por pessoa, mas o país só tem 1,3 hectares produtivos per capita.";
        //String t = "Portugal é dos países com maior consumo per capita de peixe no mundo.";
        //String h = "O oeste da Arábia Saudita vem sendo atingido por grandes tempestades de areia nos últimos dias.";
        //String t = "Cento e sete pessoas morreram na Grande Mesquita de Meca, na Arábia Saudita.";
        //String h = "Em Raanana, duas pessoas foram feridas com arma branca por um palestiniano, que acabou igualmente detido.";
        //String t = "E ao final da manhã, em Telavive, um homem foi ferido igualmente com uma arma branca.";
        //String h = "O óleo de coco é a substância mais saudável para fritar.";
        //String t = "Se pensa que fritar alimentos com óleo vegetal é mais saudável do que usar manteiga, está enganado.";
        //String h = "Temos de fazer mais e melhor, jogando bom futebol, que é a única forma de ganhar.";
        //String t = "Temos argumentos nos jogadores que estão aptos para fazer um bom jogo.";
        String h = "Além de Ishan, a polícia pediu ordens de detenção de outras 11 pessoas, a maioria deles estrangeiros.";
        String t = "Além de Ishan, a polícia deu ordem de prisão para outras 11 pessoas, a maioria estrangeiros.";
        //String h = "Mas a OMS alerta também para os milhões de pessoas, ainda por diagnosticar, que devem ser tidas em conta.";
        //String t = "De acordo com os números da OMS, há 37 milhões de pessoas em todo o mundo com VIH.";
        Par par = new Par(1, "None", 0, h, t);

        double[] cont = contador.contagem(par);
        for(double d : cont)
        	System.out.print(d+"\t");
	}*/
}
